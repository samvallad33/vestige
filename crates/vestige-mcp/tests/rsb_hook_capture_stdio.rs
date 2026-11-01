//! Retroactive Salience Backfill from memory-hook captures. Every payload
//! below is byte-for-byte what packages/vestige-init/hooks/memory.cjs sends
//! (pinned by test/memory-rsb-shape.test.js). All data is disposable.
use serde_json::{Value, json};
use std::io::{BufRead, BufReader, Write};
use std::path::Path;
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};
use std::time::Duration;

struct Server {
    child: Child,
    input: ChildStdin,
    output: BufReader<ChildStdout>,
    seq: u64,
}
impl Server {
    fn open(path: &Path, autofire: &str) -> Self {
        let mut child = Command::new(env!("CARGO_BIN_EXE_vestige-mcp"))
            .args(["--no-http", "--data-dir"])
            .arg(path)
            .env("VESTIGE_BACKFILL_AUTOFIRE", autofire)
            .env("VESTIGE_DASHBOARD_ENABLED", "false")
            .env("VESTIGE_HTTP_ENABLED", "0")
            .env("VESTIGE_AUTOPILOT_ENABLED", "0")
            .env("VESTIGE_TRACE_ENABLED", "1")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .unwrap();
        let input = child.stdin.take().unwrap();
        let output = BufReader::new(child.stdout.take().unwrap());
        let mut server = Self {
            child,
            input,
            output,
            seq: 0,
        };
        server.rpc("initialize",json!({"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"rsb-acceptance","version":"1"}}));
        server
    }
    fn rpc(&mut self, method: &str, params: Value) -> Value {
        self.seq += 1;
        writeln!(
            self.input,
            "{}",
            json!({"jsonrpc":"2.0","id":self.seq,"method":method,"params":params})
        )
        .unwrap();
        self.input.flush().unwrap();
        loop {
            let mut line = String::new();
            assert!(
                self.output.read_line(&mut line).unwrap() > 0,
                "server closed"
            );
            let response: Value = serde_json::from_str(&line).unwrap();
            if response["id"] == self.seq {
                assert!(response.get("error").is_none(), "{response}");
                return response["result"].clone();
            }
        }
    }
    fn tool(&mut self, name: &str, args: Value) -> Value {
        let result = self.rpc("tools/call", json!({"name":name,"arguments":args}));
        assert_ne!(result["isError"], true, "{name}: {result}");
        result.get("structuredContent").cloned().unwrap_or_else(|| {
            serde_json::from_str(result["content"][0]["text"].as_str().unwrap()).unwrap()
        })
    }
    fn get(&mut self, id: &str) -> Value {
        self.tool("memory", json!({"action":"get","id":id}))["node"].clone()
    }
}
impl Drop for Server {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

impl Server {
    fn hook(&mut self, item: Value) -> String {
        std::thread::sleep(Duration::from_millis(3));
        let saved = self.tool("smart_ingest", item);
        saved
            .get("nodeId")
            .or_else(|| saved.get("echoNodeId"))
            .and_then(Value::as_str)
            .unwrap_or_else(|| panic!("missing node: {saved}"))
            .to_string()
    }
    fn stability(&mut self, id: &str) -> f64 {
        self.get(id)["stability"].as_f64().unwrap()
    }
    fn receipt(&mut self, memory: &str, trigger: &str) -> Value {
        let receipts = self.tool("backfill", json!({"action":"receipts"}));
        receipts["receipts"]
            .as_array()
            .unwrap()
            .iter()
            .find(|r| r["memory_id"] == memory && r["trigger_id"] == trigger)
            .cloned()
            .unwrap_or_else(|| panic!("no receipt {memory} <- {trigger}: {receipts}"))
    }
}

fn hook_fact(content: &str, entities: Value, attributes: Value, tags: Value) -> Value {
    json!({"content":content,"entities":entities,"attributes":attributes,"node_type":"fact",
           "source":"memory-hook:verbatim-user-statement","tags":tags})
}

#[test]
fn hook_captured_health_fact_raises_the_older_reaction_through_one_person_tag() {
    let dir = tempfile::tempdir().unwrap();
    let mut s = Server::open(dir.path(), "on");
    let reaction = s.hook(hook_fact(
        "Maya felt off after dinner at Bangkok Garden",
        json!(["person:Maya", "place:Bangkok Garden"]),
        json!(["health:off"]),
        json!([
            "memory-hook-utterance",
            "slot:health",
            "situation:food-meal",
            "situation:outdoors",
            "facts-of:Maya"
        ]),
    ));
    s.tool("memory", json!({"action":"demote","id":reaction}));
    let before = s.stability(&reaction);
    let allergy = s.hook(hook_fact(
        "Maya has a peanut allergy",
        json!(["person:Maya"]),
        json!(["allergy:peanut"]),
        json!([
            "memory-hook-utterance",
            "slot:health",
            "situation:health",
            "facts-of:Maya",
            "avoid:peanut"
        ]),
    ));
    let tags = s.get(&allergy)["tags"].clone();
    let people: Vec<&str> = tags
        .as_array()
        .unwrap()
        .iter()
        .filter_map(Value::as_str)
        .filter(|t| t.starts_with("person:"))
        .collect();
    assert_eq!(people, ["person:Maya"], "exactly one person: tag, {tags}");
    assert!(
        s.stability(&reaction) > before,
        "RSB must raise the reaction"
    );
    let receipt = s.receipt(&reaction, &allergy);
    assert_eq!(receipt["shared_entity"], "person:Maya");
    assert!(
        receipt["reason"]
            .as_str()
            .unwrap()
            .contains("new health fact about Maya")
    );
    // The hook packet serves the raised memory first.
    let packet = s.tool(
        "session_start",
        json!({"memory_hook":{"prompt":"where should we take Maya for her birthday dinner?","limit":8}}),
    );
    assert_eq!(packet["memories"][0]["id"], reaction, "{packet}");
}

#[test]
fn hook_captured_conflicting_preference_lowers_the_old_value() {
    let dir = tempfile::tempdir().unwrap();
    let mut s = Server::open(dir.path(), "on");
    let old = s.hook(hook_fact(
        "Maya loves spicy food",
        json!(["person:Maya"]),
        json!(["preference:spicy food=like"]),
        json!(["memory-hook-utterance", "slot:preference", "facts-of:Maya"]),
    ));
    let before = s.stability(&old);
    let new = s.hook(hook_fact(
        "Maya hates spicy food",
        json!(["person:Maya"]),
        json!(["preference:spicy food=dislike"]),
        json!(["memory-hook-utterance", "slot:preference", "facts-of:Maya"]),
    ));
    let after = s.stability(&old);
    assert!(
        after < before && after + 1e-8 >= before / 1.5,
        "{before} -> {after}"
    );
    let receipt = s.receipt(&old, &new);
    assert!(receipt["new_stability_q"].as_i64() < receipt["old_stability_q"].as_i64());
    // Another person's same preference is never touched.
    let noah = s.hook(hook_fact(
        "Noah loves spicy food",
        json!(["person:Noah"]),
        json!(["preference:spicy food=like"]),
        json!(["memory-hook-utterance", "slot:preference", "facts-of:Noah"]),
    ));
    let noah_before = s.stability(&noah);
    s.hook(hook_fact(
        "Maya hates spicy food again",
        json!(["person:Maya"]),
        json!(["preference:spicy food=dislike"]),
        json!(["memory-hook-utterance", "slot:preference", "facts-of:Maya"]),
    ));
    assert_eq!(s.stability(&noah), noah_before);
}

#[test]
fn hook_captured_move_supersedes_and_lowers_the_old_residence() {
    let dir = tempfile::tempdir().unwrap();
    let mut s = Server::open(dir.path(), "on");
    let austin = s.hook(hook_fact(
        "Maya lives in Austin",
        json!(["person:Maya", "place:Austin"]),
        json!(["preference:residence=Austin"]),
        json!([
            "memory-hook-utterance",
            "slot:location",
            "slot:maya:residence",
            "facts-of:Maya"
        ]),
    ));
    let before = s.stability(&austin);
    let mut denver = hook_fact(
        "Maya moved to Denver",
        json!(["person:Maya", "place:Denver"]),
        json!(["preference:residence=Denver"]),
        json!([
            "memory-hook-utterance",
            "slot:location",
            "slot:maya:residence",
            "facts-of:Maya"
        ]),
    );
    // memory.cjs linksFor: same slot key, new content.
    denver["links"] = json!([{"kind":"supersedes","to":austin}]);
    std::thread::sleep(Duration::from_millis(3));
    let saved = s.tool("smart_ingest", denver);
    let denver = saved["nodeId"].as_str().unwrap().to_string();
    assert_eq!(saved["links"][0]["kind"], "supersedes", "{saved}");
    assert!(
        s.stability(&austin) < before,
        "conflicting residence lowers Austin"
    );
    s.receipt(&austin, &denver);
    // The supersedes link is an evidence edge, not a retirement: the old
    // record stays readable, lowered, and ranks after the new one.
    let packet = s.tool(
        "session_start",
        json!({"memory_hook":{"prompt":"where does Maya live?","limit":8}}),
    );
    assert_eq!(packet["memories"][0]["id"], denver, "{packet}");
}

#[test]
fn a_second_spelling_of_the_person_would_silence_backfill() {
    // Guard for the hook: person:Maya plus person:maya is two identities, and
    // plan_salience refuses a personal fact with more than one person.
    let dir = tempfile::tempdir().unwrap();
    let mut s = Server::open(dir.path(), "on");
    let reaction = s.hook(hook_fact(
        "Maya felt off after dinner at Bangkok Garden",
        json!(["person:Maya", "place:Bangkok Garden"]),
        json!([]),
        json!(["memory-hook-utterance"]),
    ));
    s.tool("memory", json!({"action":"demote","id":reaction}));
    let before = s.stability(&reaction);
    s.hook(hook_fact(
        "Maya has a peanut allergy",
        json!(["person:Maya"]),
        json!(["allergy:peanut"]),
        json!(["memory-hook-utterance", "person:maya"]),
    ));
    assert_eq!(s.stability(&reaction), before);
}

#[test]
fn hook_index_tags_join_a_thing_and_a_yearly_date_from_the_prompt() {
    let dir = tempfile::tempdir().unwrap();
    let mut s = Server::open(dir.path(), "on");
    let tub = s.hook(hook_fact(
        "Noah can't have hot tubs.",
        json!(["person:Noah"]),
        json!([]),
        json!([
            "memory-hook-utterance",
            "slot:avoid",
            "facts-of:Noah",
            "avoid:hot-tub"
        ]),
    ));
    let birthday = s.hook(hook_fact(
        "Mia's birthday is March 3",
        json!(["person:Mia"]),
        json!(["preference:birthday=03-03"]),
        json!([
            "memory-hook-utterance",
            "slot:date",
            "slot:mia:birthday",
            "facts-of:Mia",
            "yearly:03-03"
        ]),
    ));
    let spa = s.tool(
        "session_start",
        json!({"memory_hook":{"prompt":"book the spa with the Hot Tub for the weekend","limit":8}}),
    );
    assert!(
        spa["memories"]
            .as_array()
            .unwrap()
            .iter()
            .any(|n| n["id"] == tub),
        "{spa}"
    );
    let date = s.tool(
        "session_start",
        json!({"memory_hook":{"prompt":"plans for march 3rd?","limit":8}}),
    );
    assert!(
        date["memories"]
            .as_array()
            .unwrap()
            .iter()
            .any(|n| n["id"] == birthday),
        "{date}"
    );
}

#[test]
fn a_date_joined_memory_is_reserved_ahead_of_busier_lexical_hits() {
    let dir = tempfile::tempdir().unwrap();
    let mut s = Server::open(dir.path(), "on");
    let dentist = s.hook(hook_fact(
        "Dentist appointment for Sam",
        json!([]),
        json!([]),
        json!(["memory-hook-utterance", "day:2026-11-03", "yearly:11-03"]),
    ));
    for i in 0..5 {
        s.hook(hook_fact(
            &format!("Weekend sauna spa trip packing list number {i}: sauna towels, spa sandals"),
            json!([]),
            json!([]),
            json!(["memory-hook-utterance"]),
        ));
    }
    let packet = s.tool(
        "session_start",
        json!({"memory_hook":{"prompt":"sauna spa trip packing on 2026-11-03","limit":2}}),
    );
    assert_eq!(packet["memories"][0]["id"], dentist, "{packet}");
}

#[test]
fn a_named_persons_safety_fact_is_reserved_ahead_of_busier_lexical_hits() {
    let dir = tempfile::tempdir().unwrap();
    let mut s = Server::open(dir.path(), "on");
    let allergy = s.hook(hook_fact(
        "Maya has a peanut allergy",
        json!(["person:Maya"]),
        json!(["allergy:peanut"]),
        json!([
            "memory-hook-utterance",
            "slot:health",
            "facts-of:Maya",
            "avoid:peanut"
        ]),
    ));
    for i in 0..5 {
        s.hook(hook_fact(
            &format!("Maya dinner plans Thai restaurant downtown idea {i}"),
            json!(["person:Maya"]),
            json!([]),
            json!(["memory-hook-utterance"]),
        ));
    }
    let packet = s.tool(
        "session_start",
        json!({"memory_hook":{"prompt":"dinner plans with Maya at a Thai restaurant downtown","limit":2}}),
    );
    assert_eq!(packet["memories"][0]["id"], allergy, "{packet}");
}

#[test]
fn an_import_batch_is_indexed_oldest_first_and_keeps_each_original_moment() {
    let dir = tempfile::tempdir().unwrap();
    let mut s = Server::open(dir.path(), "off");
    let item = |content: &str, from: Option<&str>| {
        let mut v = json!({"content":content,"node_type":"fact","tags":["imported"]});
        if let Some(from) = from {
            v["validFrom"] = json!(from);
        }
        v
    };
    let saved = s.tool(
        "smart_ingest",
        json!({"items":[
            item("Undated import line about the garden", None),
            item("Ana started the new job in Lisbon", Some("2024-06-01T12:00:00Z")),
            item("Ana graduated from university in Porto", Some("2019-07-15T12:00:00Z")),
        ]}),
    );
    let ids: Vec<String> = (0..3)
        .map(|i| {
            saved["results"]
                .as_array()
                .unwrap()
                .iter()
                .find(|r| r["index"] == i)
                .and_then(|r| r["nodeId"].as_str())
                .unwrap_or_else(|| panic!("{saved}"))
                .to_string()
        })
        .collect();
    let created = |s: &mut Server, id: &str| s.get(id)["createdAt"].as_str().unwrap().to_string();
    let (undated, job, school) = (
        created(&mut s, &ids[0]),
        created(&mut s, &ids[1]),
        created(&mut s, &ids[2]),
    );
    assert!(
        school <= job && job <= undated,
        "oldest first: {school} {job} {undated}"
    );
    let school_row = saved["results"]
        .as_array()
        .unwrap()
        .iter()
        .find(|r| r["index"] == 2)
        .unwrap()
        .to_string();
    assert!(
        school_row.contains("2019-07-15"),
        "keeps the original moment: {school_row}"
    );
}
