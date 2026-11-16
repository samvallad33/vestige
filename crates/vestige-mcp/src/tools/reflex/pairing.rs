//! Failure Pairing. A failed call is saved once as a memory tagged with its
//! signature, linked to the fix when the same command next succeeds, and the
//! agent hears about it only when history already holds the answer: an earlier
//! failure with the same signature that has a recorded fix, told at once, or a
//! linked lesson that has faded, found in the background and told at the next
//! firing. Probes that fail by design, interrupts, refusals and gate stops are
//! never recorded. No model is involved anywhere in this path.

use super::*;

/// Fixes remembered by signature in reflex's own state.
pub(super) const MAX_FIXES: usize = 1000;
/// Open failures kept per session, waiting for the command that fixes them.
const OPEN_KEEP: usize = 32;
/// Files remembered per open failure as edited before the fix.
const EDITED_KEEP: usize = 20;
/// A linked lesson below this retention has faded enough to mention.
const FADED_PCT: f64 = 50.0;

/// A fix reflex recorded for a failure signature.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub(super) struct Fix {
    pub failure: String,
    pub fix: String,
    pub command: String,
    pub files: Vec<String>,
    pub ts: String,
}

/// A failure of this session still waiting for the call that fixes it.
#[derive(Clone, Debug, Default)]
pub(super) struct OpenFailure {
    pub sig: String,
    pub node: String,
    pub tool: String,
    pub words: String,
    pub agent: String,
    pub edited: Vec<String>,
}

/// What one failed call says about itself.
#[derive(Clone, Debug, PartialEq)]
pub(super) struct Failure {
    pub tool: String,
    pub words: String,
    pub exit: Option<i64>,
    pub line: String,
    pub sig: String,
}

pub(super) fn is_shell(tool: &str) -> bool {
    matches!(
        tool,
        "Bash" | "PowerShell" | "shell" | "exec_command" | "run_shell_command"
    )
}

/// The command reflex attributes the call to: the first segment of a chain
/// that is not a directory change or a variable assignment.
fn segment(command: &str) -> Vec<String> {
    for part in command.split(['&', '|', ';', '\n']) {
        let words: Vec<String> = part
            .split_whitespace()
            .skip_while(|w| w.contains('=') && !w.starts_with('-') && !w.contains('/'))
            .map(str::to_string)
            .collect();
        match words.first().map(String::as_str) {
            None | Some("cd" | "pushd" | "export" | "set" | "source" | ".") => continue,
            Some(_) => return words,
        }
    }
    Vec::new()
}

fn program(word: &str) -> String {
    word.rsplit('/').next().unwrap_or(word).to_string()
}

/// The first words that name what was run: the program and its first
/// argument for a shell command, the tool name for anything else.
pub(super) fn words_of(tool: &str, command: &str) -> String {
    if !is_shell(tool) {
        return tool.to_string();
    }
    let seg = segment(command);
    let mut out: Vec<String> = Vec::new();
    if let Some(first) = seg.first() {
        out.push(program(first));
    }
    if let Some(second) = seg.get(1)
        && !second.starts_with('-')
    {
        out.push(second.clone());
    }
    out.join(" ")
}

/// Paths, numbers and hashes vary between two runs of the same failure, so
/// they are masked before the signature is taken.
pub(super) fn mask(line: &str) -> String {
    line.split_whitespace()
        .map(|w| {
            if w.contains('/') || w.starts_with('~') {
                "<path>".to_string()
            } else if w.len() >= 7 && w.chars().all(|c| c.is_ascii_hexdigit()) {
                "<hex>".to_string()
            } else {
                w.chars()
                    .map(|c| if c.is_ascii_digit() { '#' } else { c })
                    .collect()
            }
        })
        .collect::<Vec<_>>()
        .join(" ")
}

/// Classify a failed call by a fixed table. `None` means it is not a failure
/// worth remembering: an interrupt, a refusal, a gate stop, or a probe that
/// answers by failing.
pub(super) fn classify(tool: &str, command: &str, error: &str, interrupt: bool) -> Option<Failure> {
    if interrupt || tool.is_empty() {
        return None;
    }
    let lower = error.to_lowercase();
    let refused = [
        "doesn't want to proceed",
        "user rejected",
        "request interrupted",
        "permission denied by user",
    ];
    let stopped = ["hook error", "operator: stopped", "blocked by hook"];
    if refused
        .iter()
        .chain(stopped.iter())
        .any(|m| lower.contains(m))
    {
        return None;
    }
    let mut exit = None;
    let mut line = String::new();
    for l in error.lines().map(str::trim).filter(|l| !l.is_empty()) {
        if let Some(n) = l.strip_prefix("Exit code ") {
            exit = n.trim().parse::<i64>().ok();
            continue;
        }
        if l.contains("characters truncated]") {
            continue;
        }
        line = l.chars().take(200).collect();
        break;
    }
    let words = words_of(tool, command);
    if is_shell(tool) {
        let seg = segment(command);
        let prog = seg.first().map(|w| program(w)).unwrap_or_default();
        let missing = lower.contains("no such file") || lower.contains("not found");
        let probe = match prog.as_str() {
            "ls" | "test" | "[" | "stat" | "which" | "type" | "command" | "file" | "readlink"
            | "realpath" => missing || exit == Some(1),
            "grep" | "egrep" | "fgrep" | "rg" | "ag" | "ack" | "pgrep" | "pidof" => exit == Some(1),
            "diff" | "cmp" => exit == Some(1),
            "git" => {
                let sub = seg.get(1).map(String::as_str).unwrap_or("");
                exit == Some(1)
                    && (sub == "grep"
                        || (sub == "diff"
                            && (command.contains("--quiet") || command.contains("--exit-code"))))
            }
            _ => false,
        };
        if probe {
            return None;
        }
    } else if tool == "Read" && lower.contains("does not exist") {
        return None;
    }
    if line.is_empty() {
        line = exit
            .map(|e| format!("exit {e}"))
            .unwrap_or_else(|| "failed".into());
    }
    let sig = sha256_hex(&format!("{tool}\n{words}\n{}", mask(&line)))[..16].to_string();
    Some(Failure {
        tool: tool.to_string(),
        words,
        exit,
        line,
        sig,
    })
}

/// Hide what looks like a credential before a command is saved.
pub(super) fn redact(command: &str) -> String {
    let mut after_bearer = false;
    command
        .split(' ')
        .map(|w| {
            let lower = w.to_lowercase();
            let hide = after_bearer
                || [
                    "token=",
                    "secret=",
                    "password=",
                    "passwd=",
                    "apikey=",
                    "api_key=",
                    "key=",
                ]
                .iter()
                .any(|k| lower.contains(k))
                || (w.len() >= 32
                    && w.chars()
                        .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-')
                    && w.chars().any(|c| c.is_ascii_digit())
                    && w.chars().any(|c| c.is_ascii_uppercase()));
            after_bearer = lower
                .trim_matches(|c| c == '"' || c == '\'')
                .ends_with("bearer");
            if hide {
                "<redacted>".to_string()
            } else {
                w.to_string()
            }
        })
        .collect::<Vec<_>>()
        .join(" ")
        .chars()
        .take(300)
        .collect()
}

/// The checked-out commit for a working directory, read from git's own files.
pub(super) fn git_head(cwd: &str) -> Option<String> {
    if cwd.is_empty() {
        return None;
    }
    let mut dir = PathBuf::from(cwd);
    for _ in 0..12 {
        let dotgit = dir.join(".git");
        if dotgit.is_file() {
            let text = std::fs::read_to_string(&dotgit).ok()?;
            let gitdir = text.strip_prefix("gitdir:")?.trim();
            let gitdir = if Path::new(gitdir).is_absolute() {
                PathBuf::from(gitdir)
            } else {
                dir.join(gitdir)
            };
            return head_in(&gitdir);
        }
        if dotgit.is_dir() {
            return head_in(&dotgit);
        }
        if !dir.pop() {
            return None;
        }
    }
    None
}

/// The repository root for a working directory, read from git's own files.
pub(super) fn git_root(cwd: &str) -> Option<String> {
    if cwd.is_empty() {
        return None;
    }
    let mut dir = PathBuf::from(cwd);
    for _ in 0..12 {
        let dotgit = dir.join(".git");
        if dotgit.is_file() || dotgit.is_dir() {
            return Some(dir.to_string_lossy().to_string());
        }
        if !dir.pop() {
            return None;
        }
    }
    None
}

fn head_in(gitdir: &Path) -> Option<String> {
    let head = std::fs::read_to_string(gitdir.join("HEAD")).ok()?;
    let head = head.trim();
    let sha = match head.strip_prefix("ref:") {
        None => head.to_string(),
        Some(r) => {
            let r = r.trim();
            let common = std::fs::read_to_string(gitdir.join("commondir"))
                .ok()
                .map(|c| gitdir.join(c.trim()))
                .unwrap_or_else(|| gitdir.to_path_buf());
            std::fs::read_to_string(gitdir.join(r))
                .or_else(|_| std::fs::read_to_string(common.join(r)))
                .map(|s| s.trim().to_string())
                .ok()
                .or_else(|| {
                    let packed = std::fs::read_to_string(common.join("packed-refs")).ok()?;
                    packed
                        .lines()
                        .find(|l| l.ends_with(r))
                        .and_then(|l| l.split_whitespace().next())
                        .map(str::to_string)
                })?
        }
    };
    (sha.len() >= 12 && is_hex(&sha)).then(|| sha[..12].to_string())
}

fn pairing_tags(inp: &Input, kind: &str, sig: &str) -> Vec<String> {
    let mut tags = vec![
        format!("{kind}-{sig}"),
        if kind == "fail" {
            "reflex-failure".to_string()
        } else {
            "reflex-fix".to_string()
        },
        session_tag(&inp.session),
    ];
    if let Some(root) = git_root(&inp.cwd) {
        tags.push(format!("repo:{root}"));
    }
    if !inp.agent.is_empty() {
        tags.push(format!("agent:{}", inp.agent));
    }
    tags
}

fn failure_item(
    inp: &Input,
    f: &Failure,
    last_write: Option<&str>,
    reopen_of: Option<&str>,
) -> Value {
    let mut content = format!(
        "TOOL FAILURE, recorded by Reflex. When: {} UTC. Session {}{}. Tool: {}. Ran: {}. \
         Error: {}. Exit code: {}. Working directory: {}. Git HEAD: {}. Call {}. Signature fail-{}: \
         the same failure again carries the same tag, so recall(handle='fail-{}') lists every time it happened.",
        Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Secs, true),
        or_unknown(&inp.session.chars().take(8).collect::<String>()),
        if inp.agent.is_empty() {
            String::new()
        } else {
            format!(", subagent {}", inp.agent)
        },
        f.tool,
        if inp.command.is_empty() {
            f.words.clone()
        } else {
            format!("`{}`", redact(&inp.command))
        },
        f.line,
        f.exit.map_or_else(|| "none".to_string(), |e| e.to_string()),
        or_unknown(&inp.cwd),
        git_head(&inp.cwd).unwrap_or_else(|| "none".into()),
        or_unknown(&inp.call),
        f.sig,
        f.sig,
    );
    let mut links = Vec::new();
    if let Some(w) = last_write {
        content.push_str(&format!(
            " Derived from {w}, the last memory this session wrote."
        ));
        links.push(json!({"kind": "derived_from", "to": w}));
    }
    if let Some(fix) = reopen_of {
        content.push_str(&format!(
            " Reopened the fix {fix} on the same exact handle fail-{} after a later failure.",
            f.sig
        ));
        links.push(json!({"kind": "reopened", "to": fix}));
    }
    json!({
        "node_type": "event",
        "tags": pairing_tags(inp, "fail", &f.sig),
        "source": format!("reflex {VERSION}, PostToolUseFailure {}", inp.call),
        "links": links,
        "content": content,
    })
}

fn fix_item(inp: &Input, open: &OpenFailure) -> Value {
    let files = if open.edited.is_empty() {
        "none, so the same command simply succeeded on a later try".to_string()
    } else {
        open.edited.join(", ")
    };
    json!({
        "node_type": "event",
        "tags": pairing_tags(inp, "fix", &open.sig),
        "source": format!("reflex {VERSION}, PostToolUse {}", inp.call),
        "links": [{"kind": "closes", "to": open.node}],
        "content": format!(
            "FIX, recorded by Reflex. When: {} UTC. The failure {} (signature fail-{}) was followed by \
             the same {} call succeeding in session {}. Command that worked: {}. Files edited in between: {}.",
            Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Secs, true),
            open.node,
            open.sig,
            open.words,
            or_unknown(&inp.session.chars().take(8).collect::<String>()),
            if inp.command.is_empty() {
                open.words.clone()
            } else {
                format!("`{}`", redact(&inp.command))
            },
            files,
        ),
    })
}

/// The fix recorded for an earlier failure carrying this signature, read
/// from the store: a failure memory tagged `fail-<sig>` with a `closed_by`
/// edge to its fix.
async fn stored_fix(ctx: &Ctx<'_>, sig: &str) -> Option<(String, String, String)> {
    let out = super::super::recall::execute(
        ctx.storage,
        ctx.cognitive,
        ctx.output_config,
        Some(json!({"handle": format!("fail-{sig}"), "detail_level": "summary", "limit": 20})),
    )
    .await
    .ok()?;
    let neighbors = out.get("neighbors").and_then(Value::as_array)?;
    neighbors.iter().find_map(|n| {
        (n.get("link_type").and_then(Value::as_str) == Some("closed_by")).then(|| {
            let failure = n
                .get("from")
                .and_then(Value::as_str)
                .unwrap_or("")
                .to_string();
            let fix = n
                .get("to")
                .and_then(Value::as_str)
                .unwrap_or("")
                .to_string();
            let content = n
                .pointer("/node/content")
                .and_then(Value::as_str)
                .unwrap_or("")
                .chars()
                .take(220)
                .collect::<String>();
            (failure, fix, content)
        })
    })
}

/// What this firing recorded, for the ledger, and what to tell the agent.
#[derive(Default)]
pub(super) struct Paired {
    pub wrote: Vec<Value>,
    pub tell: Vec<String>,
}

/// Failure Pairing for one firing: record a failure, pair a success with
/// the failure it fixes, and remember files edited in between.
pub(super) async fn on_event(
    global: &Arc<std::sync::Mutex<Global>>,
    ctx: &Ctx<'_>,
    inp: &Input,
) -> Paired {
    let mut out = Paired::default();
    if inp.session.is_empty() {
        return out;
    }
    match inp.event.as_str() {
        "PostToolUseFailure" => {
            let Some(f) = classify(&inp.tool, &inp.command, &inp.error, inp.interrupt) else {
                return out;
            };
            let (known, last_write) = {
                let Ok(mut g) = global.lock() else {
                    return out;
                };
                let known = lookup(&g.p.fixes, &f.sig).cloned();
                let last = g.session(&inp.session).last_write.clone();
                (known, last)
            };
            let earlier = match known {
                Some(fx) => Some((
                    fx.failure,
                    fx.fix,
                    format!("Command that worked: `{}`", fx.command),
                )),
                None => stored_fix(ctx, &f.sig).await,
            };
            let reopen_of = earlier.as_ref().map(|(_, fix, _)| fix.clone());
            if let Some((failure, fix, how)) = earlier {
                out.tell.push(format!(
                    "Reflex: this exact failure happened before (signature fail-{}). Failure {failure} was fixed, \
                     recorded as {fix} through a closed_by edge. {} Try that first.",
                    f.sig,
                    how.trim_end_matches('.')
                ));
            }
            match ingest(
                ctx,
                failure_item(inp, &f, last_write.as_deref(), reopen_of.as_deref()),
            )
            .await
            {
                Ok(node) => {
                    out.wrote.push(
                        json!({"kind": "failure", "memory": node, "signature": f.sig, "new": true}),
                    );
                    if let Ok(mut g) = global.lock() {
                        let s = g.session(&inp.session);
                        s.open_failures.retain(|o| o.sig != f.sig);
                        s.open_failures.push(OpenFailure {
                            sig: f.sig.clone(),
                            node: node.clone(),
                            tool: f.tool.clone(),
                            words: f.words.clone(),
                            agent: inp.agent.clone(),
                            edited: Vec::new(),
                        });
                        while s.open_failures.len() > OPEN_KEEP {
                            s.open_failures.remove(0);
                        }
                    }
                    spawn_lessons(global, ctx, inp.session.clone(), node);
                }
                Err(e) => tracing::warn!(error = %e, "reflex could not record a tool failure"),
            }
        }
        "PostToolUse" => {
            let edits = matches!(
                inp.tool.as_str(),
                "Edit" | "Write" | "MultiEdit" | "NotebookEdit"
            );
            let open = {
                let Ok(mut g) = global.lock() else {
                    return out;
                };
                let s = g.session(&inp.session);
                if edits && !inp.file.is_empty() {
                    for o in s.open_failures.iter_mut().filter(|o| o.agent == inp.agent) {
                        if !o.edited.contains(&inp.file) && o.edited.len() < EDITED_KEEP {
                            o.edited.push(inp.file.clone());
                        }
                    }
                }
                let words = words_of(&inp.tool, &inp.command);
                let pos = s.open_failures.iter().position(|o| {
                    o.tool == inp.tool
                        && o.agent == inp.agent
                        && !words.is_empty()
                        && o.words == words
                });
                pos.map(|p| s.open_failures.remove(p))
            };
            if let Some(open) = open {
                match ingest(ctx, fix_item(inp, &open)).await {
                    Ok(fix) => {
                        out.wrote.push(json!({"kind": "fix", "memory": fix, "closes": open.node, "signature": open.sig, "new": true}));
                        if let Ok(mut g) = global.lock() {
                            upsert(
                                &mut g.p.fixes,
                                &open.sig,
                                Fix {
                                    failure: open.node.clone(),
                                    fix,
                                    command: redact(&inp.command),
                                    files: open.edited.clone(),
                                    ts: Utc::now()
                                        .to_rfc3339_opts(chrono::SecondsFormat::Secs, true),
                                },
                                MAX_FIXES,
                            );
                            g.dirty = true;
                        }
                    }
                    Err(e) => tracing::warn!(error = %e, "reflex could not record a fix"),
                }
            }
        }
        _ => {}
    }
    out
}

/// Lessons linked to a failure whose retention has faded below FADED_PCT.
pub(super) fn faded_lessons(walk: &Value) -> Vec<(String, f64, String)> {
    walk.get("forgotten_lessons")
        .and_then(Value::as_array)
        .map(|rows| {
            rows.iter()
                .filter_map(|r| {
                    let id = r.get("lesson_id").and_then(Value::as_str)?.to_string();
                    let pct = r.get("retention_pct").and_then(Value::as_f64)?;
                    let preview = r
                        .get("content_preview")
                        .and_then(Value::as_str)
                        .unwrap_or("")
                        .chars()
                        .take(160)
                        .collect::<String>();
                    (pct < FADED_PCT).then_some((id, pct, preview))
                })
                .collect()
        })
        .unwrap_or_default()
}

/// Walk back from a new failure off the hook's path. A faded lesson becomes
/// a note told at the session's next firing; the walk's causes are ledgered.
fn spawn_lessons(
    global: &Arc<std::sync::Mutex<Global>>,
    ctx: &Ctx<'_>,
    session: String,
    failure: String,
) {
    if tokio::runtime::Handle::try_current().is_err() {
        return;
    }
    let global = Arc::clone(global);
    let storage = Arc::clone(ctx.storage);
    tokio::spawn(async move {
        let started = Instant::now();
        let walk = super::super::causal_walk::execute(
            &storage,
            Some(json!({
                "start_points": [{"kind": "logged_write", "node_id": failure}],
                "promote": false,
                "scan_limit": 20,
            })),
        )
        .await
        .unwrap_or_default();
        let causes: Vec<String> = walk
            .get("causes")
            .and_then(Value::as_array)
            .map(|c| {
                c.iter()
                    .filter_map(|x| x.get("id").and_then(Value::as_str).map(str::to_string))
                    .take(5)
                    .collect()
            })
            .unwrap_or_default();
        let lessons = super::super::forgotten_lesson::execute(
            &storage,
            Some(json!({"failure_id": failure, "scan_limit": 20})),
        )
        .await
        .unwrap_or_default();
        let faded = faded_lessons(&lessons);
        let Ok(mut g) = global.lock() else {
            return;
        };
        let s = g.session(&session);
        for (id, pct, preview) in &faded {
            s.notes.push(format!(
                "Reflex: the failure {failure} is linked to a lesson that has faded to {pct:.0}%: {id}, {preview}. \
                 recall(handle='{id}') reads it."
            ));
        }
        let inp = Input {
            event: "FailureWalk".into(),
            session: session.clone(),
            host: "reflex".into(),
            ..Input::default()
        };
        let faded_ids: Vec<&String> = faded.iter().map(|(id, _, _)| id).collect();
        append_ledger(
            &mut g,
            &inp,
            0,
            &[],
            &[],
            started.elapsed().as_millis(),
            json!({"failure": failure, "causes": causes, "fadedLessons": faded_ids}),
        );
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn probes_interrupts_refusals_and_gate_stops_are_not_failures() {
        let none = [
            (
                "Bash",
                "ls ~/nowhere",
                "Exit code 1\nls: ~/nowhere: No such file or directory",
                false,
            ),
            ("Bash", "grep -rn foo src", "Exit code 1", false),
            ("Bash", "rg needle .", "Exit code 1", false),
            ("Bash", "git diff --quiet", "Exit code 1", false),
            ("Bash", "test -f x", "Exit code 1", false),
            ("Bash", "cargo test", "Exit code 101\nfailed", true),
            (
                "Bash",
                "rm x",
                "The user doesn't want to proceed with this tool use.",
                false,
            ),
            (
                "Bash",
                "grep x",
                "PreToolUse:Bash hook error: OPERATOR: STOPPED BY YOUR LAW",
                false,
            ),
            ("Read", "", "File does not exist.", false),
        ];
        for (tool, cmd, err, interrupt) in none {
            assert_eq!(classify(tool, cmd, err, interrupt), None, "{tool} {cmd}");
        }
        assert!(
            classify(
                "Bash",
                "grep -rn foo src",
                "Exit code 2\ngrep: bad option",
                false
            )
            .is_some()
        );
        assert!(
            classify(
                "Bash",
                "cargo test -p x",
                "Exit code 101\nerror[E0499]: cannot borrow",
                false
            )
            .is_some()
        );
    }

    #[test]
    fn the_signature_ignores_paths_numbers_and_hashes() {
        let a = classify(
            "Bash",
            "cd /a && cargo test -p one",
            "Exit code 101\nerror at /srv/app/src/lib.rs:12 in 3f9a2b1c",
            false,
        )
        .unwrap();
        let b = classify(
            "Bash",
            "cargo test -p two",
            "Exit code 101\nerror at /tmp/other/main.rs:907 in 77aa00ff",
            false,
        )
        .unwrap();
        assert_eq!(a.words, "cargo test");
        assert_eq!(a.sig, b.sig);
        let c = classify(
            "Bash",
            "cargo build",
            "Exit code 101\nerror at /x.rs:1 in 3f9a2b1c",
            false,
        )
        .unwrap();
        assert_ne!(a.sig, c.sig, "a different command is a different failure");
    }

    #[test]
    fn credentials_never_reach_the_saved_command() {
        let r =
            redact("curl -H Authorization: Bearer abcDEF123 https://x.example API_KEY=s3cret run");
        assert!(!r.contains("abcDEF123") && !r.contains("s3cret"), "{r}");
        assert!(r.contains("curl"));
    }

    #[test]
    fn faded_lessons_are_the_ones_below_half_retention() {
        let walk = json!({"forgotten_lessons": [
            {"lesson_id": "mem-0000000000000001", "retention_pct": 31.5, "content_preview": "old fix"},
            {"lesson_id": "mem-0000000000000002", "retention_pct": 99.0, "content_preview": "fresh"}
        ]});
        let faded = faded_lessons(&walk);
        assert_eq!(faded.len(), 1);
        assert_eq!(faded[0].0, "mem-0000000000000001");
    }

    #[test]
    fn a_later_failure_declares_a_reopened_link_to_the_fix() {
        let inp = Input {
            session: "sess-1".into(),
            cwd: "/repo".into(),
            call: "c1".into(),
            command: "cargo test -p x".into(),
            agent: "worker".into(),
            ..Input::default()
        };
        let f = classify(
            "Bash",
            "cargo test -p x",
            "Exit code 101\nerror[E0499]: cannot borrow",
            false,
        )
        .unwrap();
        let first = failure_item(&inp, &f, None, None);
        let links = first["links"].as_array().unwrap();
        assert!(
            links.iter().all(|link| link["kind"] != "reopened"),
            "{first}"
        );
        let later = failure_item(&inp, &f, None, Some("mem-0000000000000002"));
        let links = later["links"].as_array().unwrap();
        assert!(
            links
                .iter()
                .any(|link| { link["kind"] == "reopened" && link["to"] == "mem-0000000000000002" }),
            "{later}"
        );
        assert!(later["content"].as_str().unwrap().contains("fail-"));
        let tags = later["tags"].as_array().unwrap();
        assert!(
            tags.iter()
                .any(|tag| tag.as_str().unwrap().starts_with("fail-"))
        );
        assert!(tags.iter().any(|tag| tag == "agent:worker"));
    }
}

/// Save-gate session (#774). Nested so `OpenFailure` here does not collide
/// with the failure-pairing `OpenFailure` above (`sig`, `node`, `tool`,
/// `words`, `agent`, `edited`).
pub mod save_gate {
    use std::collections::{BTreeMap, BTreeSet};

    use serde::{Deserialize, Serialize};
    use vestige_core::chain_fold::Hash32;
    use vestige_core::provenance::{CausalEdge, TouchedFileHunk};

    /// An active unclosed execution failure observed in the current session.
    #[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
    pub struct OpenFailure {
        pub failure_frame_digest: Hash32,
        pub command: String,
        pub error_sig: String,
        pub exit_code: i32,
        pub tool_call_digest: Hash32,
        pub git_head_sha: String,
        pub files: Vec<TouchedFileHunk>,
        pub turn_seq: u64,
        pub ts_ms: u64,
    }

    /// A failure paired with a subsequent passing command (`OpenFailure -> Fix`).
    #[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
    pub struct ResolvedFailurePair {
        pub failure: OpenFailure,
        pub fix_command: String,
        pub fix_frame_digest: Hash32,
        pub resolved_turn_seq: u64,
        pub resolved_ts_ms: u64,
        pub anchored_by_remember: bool,
    }

    /// Lightweight event anchor captured unconditionally on every `Edit`/`Write` or `git commit`.
    #[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
    pub struct FileAnchorEvent {
        pub frame_digest: Hash32,
        pub kind: String,
        pub git_head_sha: String,
        pub files: Vec<TouchedFileHunk>,
        pub turn_seq: u64,
        pub ts_ms: u64,
    }

    /// Result of checking whether a `Stop` or `PreToolUse(git commit)` must block for `remember`.
    #[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
    #[serde(tag = "decision", rename_all = "snake_case")]
    pub enum SaveGateVerdict {
        Allow,
        Block {
            missing_files: Vec<String>,
            resolved_error_sigs: Vec<String>,
            reason: String,
        },
    }

    /// Auto-bind metadata returned to `vestige-mcp` (`remember(action="save")`) when `files` or
    /// `links` are omitted by the calling agent (#774 / Wave 14M).
    #[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
    pub struct RememberAutoBindContext {
        pub session_id: String,
        pub turn_seq: u64,
        pub prompt_hash: Hash32,
        pub git_head_sha: String,
        pub bound_files: Vec<TouchedFileHunk>,
        pub parent_frame_digest: Hash32,
        pub causal_edge: CausalEdge,
        pub resolved_error_sigs: Vec<String>,
        pub rsb_target_digests: Vec<Hash32>,
    }

    /// Persistent per-session Reflex state.
    #[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
    pub struct ReflexSessionState {
        pub session_id: String,
        pub turn_seq: u64,
        pub prompt_hash: Hash32,
        pub git_head_sha: String,
        pub last_frame_digest: Hash32,
        pub edited_files_turn: BTreeMap<String, u64>,
        pub saved_files_turn: BTreeMap<String, u64>,
        pub edited_hunks: BTreeMap<String, TouchedFileHunk>,
        pub open_failures: Vec<OpenFailure>,
        pub resolved_pairs: Vec<ResolvedFailurePair>,
        pub anchor_events: Vec<FileAnchorEvent>,
        pub stop_blocks_in_turn: u32,
    }

    impl ReflexSessionState {
        pub fn new(session_id: impl Into<String>) -> Self {
            Self {
                session_id: session_id.into(),
                turn_seq: 0,
                prompt_hash: [0u8; 32],
                git_head_sha: String::new(),
                last_frame_digest: [0u8; 32],
                edited_files_turn: BTreeMap::new(),
                saved_files_turn: BTreeMap::new(),
                edited_hunks: BTreeMap::new(),
                open_failures: Vec::new(),
                resolved_pairs: Vec::new(),
                anchor_events: Vec::new(),
                stop_blocks_in_turn: 0,
            }
        }

        pub fn on_user_prompt(&mut self, prompt_hash: Hash32, git_head_sha: impl Into<String>) {
            self.turn_seq += 1;
            self.prompt_hash = prompt_hash;
            self.git_head_sha = git_head_sha.into();
            self.stop_blocks_in_turn = 0;
        }

        pub fn record_file_anchor(
            &mut self,
            kind: impl Into<String>,
            git_head_sha: impl Into<String>,
            files: Vec<TouchedFileHunk>,
            frame_digest: Hash32,
            ts_ms: u64,
        ) {
            let kind = kind.into();
            let git_head_sha = git_head_sha.into();
            self.git_head_sha = git_head_sha.clone();
            self.last_frame_digest = frame_digest;

            if kind != "git_commit" {
                for hunk in &files {
                    self.edited_files_turn
                        .insert(hunk.path.clone(), self.turn_seq);
                    self.edited_hunks.insert(hunk.path.clone(), hunk.clone());
                }
            }

            self.anchor_events.push(FileAnchorEvent {
                frame_digest,
                kind,
                git_head_sha,
                files,
                turn_seq: self.turn_seq,
                ts_ms,
            });
        }

        #[allow(clippy::too_many_arguments)]
        pub fn record_command_outcome(
            &mut self,
            command: impl Into<String>,
            exit_code: i32,
            error_sig: Option<String>,
            tool_call_digest: Hash32,
            git_head_sha: impl Into<String>,
            frame_digest: Hash32,
            ts_ms: u64,
        ) {
            let command = command.into();
            let git_head_sha = git_head_sha.into();
            self.git_head_sha = git_head_sha.clone();
            self.last_frame_digest = frame_digest;

            if exit_code != 0 {
                let files: Vec<TouchedFileHunk> = self.edited_hunks.values().cloned().collect();
                self.open_failures.push(OpenFailure {
                    failure_frame_digest: frame_digest,
                    command,
                    error_sig: error_sig.unwrap_or_else(|| format!("exit:{exit_code}")),
                    exit_code,
                    tool_call_digest,
                    git_head_sha,
                    files,
                    turn_seq: self.turn_seq,
                    ts_ms,
                });
            } else if !self.open_failures.is_empty() {
                let drained = std::mem::take(&mut self.open_failures);
                for failure in drained {
                    self.resolved_pairs.push(ResolvedFailurePair {
                        failure,
                        fix_command: command.clone(),
                        fix_frame_digest: frame_digest,
                        resolved_turn_seq: self.turn_seq,
                        resolved_ts_ms: ts_ms,
                        anchored_by_remember: false,
                    });
                }
            }
        }

        pub fn unsaved_edited_files(&self) -> Vec<String> {
            self.edited_files_turn
                .iter()
                .filter(|(path, edited_turn)| {
                    let saved_turn = self.saved_files_turn.get(*path).copied().unwrap_or(0);
                    **edited_turn > saved_turn
                })
                .map(|(path, _)| (*path).clone())
                .collect()
        }

        pub fn unanchored_resolved_errors(&self) -> Vec<String> {
            let mut seen = BTreeSet::new();
            for pair in &self.resolved_pairs {
                if !pair.anchored_by_remember {
                    seen.insert(pair.failure.error_sig.clone());
                }
            }
            seen.into_iter().collect()
        }

        pub fn check_save_gate(&self) -> SaveGateVerdict {
            let missing_files = self.unsaved_edited_files();
            let resolved_error_sigs = self.unanchored_resolved_errors();

            if missing_files.is_empty() && resolved_error_sigs.is_empty() {
                return SaveGateVerdict::Allow;
            }

            let files_desc = if missing_files.is_empty() {
                "none".to_owned()
            } else {
                missing_files.join(", ")
            };
            let errors_desc = if resolved_error_sigs.is_empty() {
                "none".to_owned()
            } else {
                resolved_error_sigs.join(", ")
            };

            let reason = format!(
                "You modified [{files_desc}] / resolved [{errors_desc}] without anchoring a \
             remember(action='save', files=..., links=...) receipt in Strata. \
             Call remember before ending your turn or committing."
            );

            SaveGateVerdict::Block {
                missing_files,
                resolved_error_sigs,
                reason,
            }
        }

        pub fn build_auto_bind_context(&self) -> RememberAutoBindContext {
            let unsaved = self.unsaved_edited_files();
            let bound_files: Vec<TouchedFileHunk> = unsaved
                .iter()
                .filter_map(|p| self.edited_hunks.get(p).cloned())
                .collect();

            let unanchored: Vec<&ResolvedFailurePair> = self
                .resolved_pairs
                .iter()
                .filter(|p| !p.anchored_by_remember)
                .collect();

            let (parent_frame_digest, causal_edge, rsb_target_digests, resolved_error_sigs) =
                if let Some(latest_pair) = unanchored.last() {
                    let mut targets = Vec::new();
                    let mut sigs = Vec::new();
                    for pair in &unanchored {
                        targets.push(pair.failure.failure_frame_digest);
                        targets.push(pair.fix_frame_digest);
                        sigs.push(pair.failure.error_sig.clone());
                    }
                    (
                        latest_pair.failure.failure_frame_digest,
                        CausalEdge::Closes,
                        targets,
                        sigs,
                    )
                } else if let Some(active_failure) = self.open_failures.last() {
                    (
                        active_failure.failure_frame_digest,
                        CausalEdge::EvidenceOf,
                        vec![active_failure.failure_frame_digest],
                        vec![active_failure.error_sig.clone()],
                    )
                } else if let Some(latest_anchor) = self.anchor_events.last() {
                    (
                        latest_anchor.frame_digest,
                        CausalEdge::DerivedFrom,
                        vec![latest_anchor.frame_digest],
                        Vec::new(),
                    )
                } else {
                    (
                        self.last_frame_digest,
                        CausalEdge::Root,
                        Vec::new(),
                        Vec::new(),
                    )
                };

            RememberAutoBindContext {
                session_id: self.session_id.clone(),
                turn_seq: self.turn_seq,
                prompt_hash: self.prompt_hash,
                git_head_sha: self.git_head_sha.clone(),
                bound_files,
                parent_frame_digest,
                causal_edge,
                resolved_error_sigs,
                rsb_target_digests,
            }
        }

        pub fn mark_remember_saved(
            &mut self,
            covered_files: &[String],
            remember_frame_digest: Hash32,
        ) {
            self.last_frame_digest = remember_frame_digest;
            for path in covered_files {
                self.saved_files_turn.insert(path.clone(), self.turn_seq);
            }
            for pair in &mut self.resolved_pairs {
                pair.anchored_by_remember = true;
            }
            self.stop_blocks_in_turn = 0;
        }
    }
}

pub use save_gate::ReflexSessionState;
