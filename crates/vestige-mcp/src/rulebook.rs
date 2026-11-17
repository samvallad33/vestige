//! Default Rulebook, version 1.
//!
//! One source for the rules every agent loads. The session-start hook, the
//! session capsule, and a Strata connect all print this text. An owner memory
//! tagged `owner-rule` whose first line is a section id (`R01` … `R17`)
//! replaces that section. Enforcement of the rules is #777; lines this tree
//! does not implement say `DEFERRED` and name the issue that builds them.

/// Version of [`TEXT`]. Bumped only when a section's meaning changes.
pub const VERSION: u32 = 1;

/// Exact tag whose memories override a section.
pub const OWNER_TAG: &str = "owner-rule";

/// Exact tag for a command whose recorded runs failed. The memory content is
/// the command line, byte for byte.
pub const RUN_FAILED_TAG: &str = "run-failed";

/// Handle for this version. Quoted back, it resolves to this text.
pub const HANDLE: &str = "vst:rulebook:1";

/// The 17 sections. Each bullet is `SHIPPED`, `PARTIAL`, or `DEFERRED:#n`.
pub const TEXT: &str = "\
VESTIGE DEFAULT RULEBOOK
version: 1
handle: `vst:rulebook:1`
Owner rules override a section when a memory tagged owner-rule names its id.

## R01 Authority
- SHIPPED: Recognized destructive calls are stopped before they run.
- SHIPPED: A risky action needs a one-use permit from the owner.
- SHIPPED: An agent never approves its own permit.
- PARTIAL: Operator fails closed only when STRICT is on, or for plainly destructive text.
- DEFERRED:#682: A subagent never gets more authority than its parent.
- SHIPPED: Writing shell init files is stopped.
- SHIPPED: Cloud metadata endpoints are stopped.

## R02 Receipts
- SHIPPED: Every stop, permit and allow gets a receipt.
- SHIPPED: Agents cannot erase their transcripts or Vestige's record.
- PARTIAL: Secrets in tool arguments are redacted in gate previews. DEFERRED:#680 for full argument redaction.

## R03 Memory provenance
- PARTIAL: Actor provenance blocks exist. DEFERRED:#525 for owner, agent, or untrusted on every memory.
- DEFERRED:#774: Every memory needs a causal anchor, or it is quarantined, never deleted.
- PARTIAL: Untrusted fetches are stored as data. DEFERRED:#526 for quarantine.

## R04 Commit discipline
- DEFERRED:#777: No save until there is a commit. The save-gate branch denies the opposite order and is not on main.
- PARTIAL: A commit receipt records the SHA and the files. Line ranges are still open.
- DEFERRED:#566: Test locally first, and show the output before anything is saved.
- DEFERRED:#776: No commit while staged files carry an unresolved contradiction or an unverified secret warning.

## R05 Shared learning
- SHIPPED: Every stop is saved to memory every agent shares.
- DEFERRED:#544: Corrections show up at the point of action.
- PARTIAL: Failed fixes can be listed. DEFERRED:#549 to show them before the next attempt.
- SHIPPED: RSB raises an old memory when new evidence shows it matters.

## R06 Truth checks
- DEFERRED:#547: A final message cannot cite a commit, file, PR or memory id that does not exist.
- DEFERRED:#566: A remembered decision counts as done only with an execution receipt.

## R07 Rule precedence
- PARTIAL: The owner's rules beat agent-written rules for the gate and for process identity.
- DEFERRED:#546: Every agent's rules file comes from one source.
- SHIPPED: Projection writes only inside a fenced region and refuses a symlink or a path outside the root.

## R08 Session start
- SHIPPED: On startup, resume, clear, compact or fork, the agent loads this rulebook.
- SHIPPED: The same load carries pinned decisions and due obligations by handle.
- PARTIAL: Active permits are re-read from gate receipts by the turnover hook.

## R09 Local and sovereign
- PARTIAL: Everything stays on the owner's machines and keys stay beside the log. Peer-to-peer sync with a merge receipt is not on main.

## R10 Proof-carrying fixes
- DEFERRED:#571: No commit without a bisect witness after a failure.
- PARTIAL: The gate stops skipped tests, suppressed lints and canned oracles. Re-running tests across commits is DEFERRED:#571.
- SHIPPED: Auto-link reverts and Fixes lines.

## R11 Loop breakers
- DEFERRED:#542: Hard stop on repeat failures.
- PARTIAL: The gate decodes obfuscated pipelines. Obfuscation itself as an escalation stop is DEFERRED:#530.
- PARTIAL: Recall is by exact handle. Forcing that over a grep loop is DEFERRED:#542.

## R12 Parallel agents
- DEFERRED:#555: Refuse destructive worktree cleanup when the target is dirty or has open intentions.
- DEFERRED:#684: Block worktrees named .git and symlink escapes.
- PARTIAL: Memories carry an exact scope. Per-worktree overwrite detection is DEFERRED:#555.

## R13 Scope discipline
- DEFERRED:#776: Contract before edits. A multi-file task names its files; edits outside need a permit.
- DEFERRED:#776: No drive-by rewrites bundled into a bugfix receipt.
- DEFERRED:#544: Landmine checks before touching a path with a prior stop.

## R14 Skeptical memory
- PARTIAL: Memory is a claim, not live ground truth. Automatic re-check is DEFERRED:#524.
- PARTIAL: Contradictions are recorded. Forcing a supersession at save time is DEFERRED:#252.
- SHIPPED: Bounded rule projection with an item cap and an FSRS retention floor.
- DEFERRED:#402: Purge leaves a tombstone. On this log, purge is withheld because the log is append-only.

## R15 Silent compaction
- SHIPPED: Re-hydrate on every compaction, not just boot: rulebook, handles, and the done ledger.
- DEFERRED:#523: Checkpoint before delegation to a subagent.

## R16 Hostile repo
- SHIPPED: Zero-trust git and repo ingestion.
- PARTIAL: Unreviewed publishes and pipe-to-shell installs are stopped. A general unpinned-install stop is open.
- PARTIAL: External issues and URLs are stored as data with an evidence hash. DEFERRED:#526 for a guarantee the prose is never an instruction.
- SHIPPED: No secret egress in MCP arguments or shell.

## R17 Safe upgrades
- SHIPPED: Non-destructive migration with a backup receipt.
- PARTIAL: Secret scan exists. Running it on migrated records is DEFERRED:#526.
- PARTIAL: Doctor, selftest and a 30-day replay exist. Running them together before enforce is open.
- DEFERRED:#776: Enforce purposeful tagging at write time. remember requires content; tags stay optional.
";

/// The rulebook as shipped, with no owner override.
pub fn default_rulebook() -> &'static str {
    TEXT
}

/// Section ids in order.
pub fn section_ids() -> Vec<String> {
    let mut ids = Vec::new();
    for line in TEXT.lines() {
        if let Some(rest) = line.strip_prefix("## ")
            && let Some(id) = rest.split_whitespace().next()
            && id.starts_with('R')
            && id.len() == 3
        {
            ids.push(id.to_string());
        }
    }
    ids
}

/// Replace the body of `id` with `body` (which should start with `## id`).
/// Unknown ids leave the text unchanged.
pub fn replace_section(text: &str, id: &str, body: &str) -> String {
    let header = format!("## {id}");
    let Some(start) = text.find(&header) else {
        return text.to_string();
    };
    let after = start + header.len();
    let end = text[after..]
        .find("\n## ")
        .map(|i| after + i)
        .unwrap_or(text.len());
    let mut out = String::with_capacity(text.len() + body.len());
    out.push_str(&text[..start]);
    let body = body.trim_end_matches('\n');
    if body.starts_with("## ") {
        out.push_str(body);
    } else {
        out.push_str(&header);
        out.push('\n');
        out.push_str(body);
    }
    out.push_str(&text[end..]);
    out
}

/// Apply owner memories. A memory counts only when its content's first line is
/// a section id. The rest of the content is that section's new body.
pub fn apply_owner<'a>(text: &str, owner: impl IntoIterator<Item = &'a str>) -> String {
    let mut out = text.to_string();
    for content in owner {
        let mut lines = content.lines();
        let Some(id) = lines.next().map(str::trim) else {
            continue;
        };
        if !section_ids().iter().any(|known| known == id) {
            continue;
        }
        let body: String = lines.collect::<Vec<_>>().join("\n");
        let section = if body.is_empty() {
            format!("## {id}\n")
        } else {
            format!("## {id}\n{body}")
        };
        out = replace_section(&out, id, &section);
    }
    out
}

/// What a session is given: the rulebook (owner sections replaced), then the
/// pinned-rule block, then rules-file flags. To stay inside `max_bytes`,
/// whole rulebook sections drop from the end first (the rulebook comes back
/// in full by its handle), then the flags, then the pinned-rule block. The
/// pinned rules are the part no handle restates, so they outlive every
/// rulebook section. The version handle is never cut.
pub fn inject(owner: &[&str], rules: Option<&str>, flags: &[String], max_bytes: usize) -> String {
    let mut book = apply_owner(TEXT, owner.iter().copied());
    let mut tail: Vec<String> = Vec::new();
    if let Some(rules) = rules.filter(|text| !text.is_empty()) {
        tail.push(rules.to_string());
    }
    if !flags.is_empty() {
        let mut block = String::from("Rules file check:");
        for flag in flags {
            block.push('\n');
            block.push_str(flag);
        }
        tail.push(block);
    }
    let compose = |book: &str, omitted: bool, tail: &[String]| {
        let mut text = book.to_string();
        if omitted {
            text.push_str("\nomitted: `");
            text.push_str(HANDLE);
            text.push('`');
        }
        for part in tail {
            text.push_str("\n\n");
            text.push_str(part);
        }
        text
    };
    let mut omitted = false;
    loop {
        let text = compose(&book, omitted, &tail);
        if text.len() <= max_bytes {
            return text;
        }
        // Only the rulebook's own sections are cut here: the later parts sit
        // after its last section and must not go with it.
        if let Some(idx) = book.rfind("\n## R") {
            book.truncate(idx);
            omitted = true;
            continue;
        }
        // No section left: the preamble shrinks to the handle line before
        // any later part goes.
        let handle_line = format!("handle: `{HANDLE}`");
        if book != handle_line {
            book = handle_line;
            omitted = false;
            continue;
        }
        if tail.pop().is_some() {
            continue;
        }
        return format!("handle: `{HANDLE}`");
    }
}

// =========================================================================
// Cycle F (#542, #544, #547, #548, #549, #555, #776, #777):
// Deterministic Reflex Enforcement & Verification Primitives
// =========================================================================

/// Recorded command failure attempt for `#542` Loop Breaker & `#549` Dead Ends.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FailureAttempt {
    pub signature: String,
    pub tree_hash: String,
    pub attempt_summary: String,
    pub receipt_id: String,
    pub resolved_fix: Option<String>,
}

/// Outcome of `#542` Loop Breaker check before running a command.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LoopBreakerDecision {
    pub blocked: bool,
    pub repeat_count: usize,
    pub recorded_fix: Option<String>,
    pub dead_ends: Vec<String>,
}

/// `#542` Loop Breaker + `#549` Dead-End Lookup:
/// Blocks a command when `signature` has already failed `>= max_retries` times with
/// the exact same `tree_hash` (no code change between tries), and surfaces any recorded
/// fix or prior failed fix attempts (`dead_ends`).
pub fn check_failure_loop(
    signature: &str,
    current_tree_hash: &str,
    prior: &[FailureAttempt],
    max_retries: usize,
) -> LoopBreakerDecision {
    let mut same_tree_count = 0usize;
    let mut recorded_fix: Option<String> = None;
    let mut dead_ends: Vec<String> = Vec::new();

    for att in prior {
        if att.signature != signature {
            continue;
        }
        if let Some(fix) = &att.resolved_fix {
            recorded_fix = Some(fix.clone());
        } else if !att.attempt_summary.is_empty() && !dead_ends.contains(&att.attempt_summary) {
            dead_ends.push(att.attempt_summary.clone());
        }
        if att.tree_hash == current_tree_hash && att.resolved_fix.is_none() {
            same_tree_count += 1;
        }
    }

    LoopBreakerDecision {
        blocked: same_tree_count >= max_retries.max(1),
        repeat_count: same_tree_count,
        recorded_fix,
        dead_ends,
    }
}

/// Anchored correction memory for `#544` Point-of-Action Corrections.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CorrectionAnchor {
    pub memory_id: String,
    pub anchor_target: String,
    pub rule_text: String,
}

/// `#544` Corrections at the Point of Action:
/// Returns all correction memories whose exact `anchor_target` matches `tool_name`,
/// `command`, or `file_path` (silence on unrelated actions).
pub fn match_action_corrections<'a>(
    tool_name: &str,
    command: Option<&str>,
    file_path: Option<&str>,
    corrections: &'a [CorrectionAnchor],
) -> Vec<&'a CorrectionAnchor> {
    corrections
        .iter()
        .filter(|c| {
            let t = c.anchor_target.trim();
            if t.is_empty() {
                return false;
            }
            t == tool_name
                || command.is_some_and(|cmd| cmd.trim() == t || cmd.contains(t))
                || file_path.is_some_and(|fp| fp.trim() == t || fp.ends_with(t))
        })
        .collect()
}

/// `#547` Done Check:
/// Verifies that every `mem-<hex>` handle, 7..40-char hex commit SHA prefix (tagged
/// `commit <sha>`), and relative file path (`path:<rel>`) cited in `final_message`
/// resolves against `known_shas`, `known_mem_ids`, and `repo_root`. Returns unresolved citations.
pub fn verify_done_citations(
    final_message: &str,
    known_shas: &[&str],
    known_mem_ids: &[&str],
    repo_root: Option<&std::path::Path>,
) -> Vec<String> {
    let mut unresolved: Vec<String> = Vec::new();
    for tok in final_message.split(|c: char| c.is_whitespace() || matches!(c, '`' | ',' | ';' | ')' | '(' | '[' | ']')) {
        let t = tok.trim_matches(|c: char| c == '.' || c == ':' || c == '"' || c == '\'');
        if t.is_empty() {
            continue;
        }
        if t.starts_with("mem-") && t.len() >= 8 {
            if !known_mem_ids.iter().any(|k| *k == t || k.starts_with(t)) {
                unresolved.push(t.to_string());
            }
        }
    }
    let words: Vec<&str> = final_message.split_whitespace().collect();
    for pair in words.windows(2) {
        if pair[0].eq_ignore_ascii_case("commit") {
            let sha = pair[1].trim_matches(|c: char| !c.is_ascii_alphanumeric());
            if (7..=40).contains(&sha.len()) && sha.bytes().all(|b| b.is_ascii_hexdigit()) {
                if !known_shas.iter().any(|k| k.starts_with(sha) || sha.starts_with(*k)) {
                    unresolved.push(sha.to_string());
                }
            }
        }
    }
    if let Some(root) = repo_root {
        for tok in final_message.split('`').skip(1).step_by(2) {
            if (tok.starts_with("crates/") || tok.starts_with("src/") || tok.starts_with("scripts/"))
                && !tok.contains(' ')
                && !root.join(tok).exists()
            {
                unresolved.push(tok.to_string());
            }
        }
    }
    unresolved.sort();
    unresolved.dedup();
    unresolved
}

/// `#548` Postmortem Pack Entry: ordered timeline record with BLAKE3 chain link.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct PostmortemEntry {
    pub seq: u64,
    pub timestamp_ms: u64,
    pub kind: String,
    pub summary: String,
    pub receipt_id: String,
    pub prev_hash_hex: String,
    pub entry_hash_hex: String,
}

fn hash_postmortem_step(
    seq: u64,
    timestamp_ms: u64,
    kind: &str,
    summary: &str,
    receipt_id: &str,
    prev_hash_hex: &str,
) -> String {
    let mut h = blake3::Hasher::new();
    h.update(&seq.to_le_bytes());
    h.update(&timestamp_ms.to_le_bytes());
    h.update(kind.as_bytes());
    h.update(&[0]);
    h.update(summary.as_bytes());
    h.update(&[0]);
    h.update(receipt_id.as_bytes());
    h.update(&[0]);
    h.update(prev_hash_hex.as_bytes());
    h.finalize().to_hex().to_string()
}

/// `#548` Build an ordered, hash-chained postmortem pack from raw `(timestamp_ms, kind, summary, receipt_id)` events.
pub fn build_postmortem_pack(events: &[(u64, &str, &str, &str)]) -> Vec<PostmortemEntry> {
    let mut sorted: Vec<(u64, &str, &str, &str)> = events.to_vec();
    sorted.sort_by_key(|e| e.0);
    let mut out = Vec::with_capacity(sorted.len());
    let mut prev = "0".repeat(64);
    for (idx, &(ts, kind, summary, receipt)) in sorted.iter().enumerate() {
        let seq = idx as u64;
        let hash = hash_postmortem_step(seq, ts, kind, summary, receipt, &prev);
        out.push(PostmortemEntry {
            seq,
            timestamp_ms: ts,
            kind: kind.to_string(),
            summary: summary.to_string(),
            receipt_id: receipt.to_string(),
            prev_hash_hex: prev,
            entry_hash_hex: hash.clone(),
        });
        prev = hash;
    }
    out
}

/// `#548` Verify offline integrity of a postmortem pack; returns `Err(seq)` on the first tampered or out-of-order entry.
pub fn verify_postmortem_pack(pack: &[PostmortemEntry]) -> Result<(), u64> {
    let mut prev = "0".repeat(64);
    let mut last_ts = 0u64;
    for (idx, e) in pack.iter().enumerate() {
        if e.seq != idx as u64 || e.timestamp_ms < last_ts || e.prev_hash_hex != prev {
            return Err(e.seq);
        }
        let expected = hash_postmortem_step(
            e.seq,
            e.timestamp_ms,
            &e.kind,
            &e.summary,
            &e.receipt_id,
            &e.prev_hash_hex,
        );
        if e.entry_hash_hex != expected {
            return Err(e.seq);
        }
        prev = e.entry_hash_hex.clone();
        last_ts = e.timestamp_ms;
    }
    Ok(())
}

/// `#555` Jepsen Elle File Access Operation for concurrent agent lost-update detection.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AgentFileWrite {
    pub agent_id: String,
    pub path: String,
    pub read_hash: String,
    pub write_hash: String,
    pub receipt_id: String,
}

/// `#555` Detect lost updates where two distinct agents read the same base `read_hash`
/// on `path` and both write different `write_hash` states without chaining (`w2.read_hash != w1.write_hash`).
pub fn detect_elle_lost_updates(writes: &[AgentFileWrite]) -> Vec<(String, String, String)> {
    let mut anomalies: Vec<(String, String, String)> = Vec::new();
    for i in 0..writes.len() {
        for j in (i + 1)..writes.len() {
            let a = &writes[i];
            let b = &writes[j];
            if a.path == b.path
                && a.agent_id != b.agent_id
                && a.read_hash == b.read_hash
                && a.write_hash != b.write_hash
            {
                anomalies.push((a.path.clone(), a.receipt_id.clone(), b.receipt_id.clone()));
            }
        }
    }
    anomalies
}

/// `#777` R04 Commit-Before-Save Gate: denies a `remember(save)` when `has_commit_receipt` is false.
pub fn enforce_save_after_commit(has_commit_receipt: bool) -> Result<(), &'static str> {
    if has_commit_receipt {
        Ok(())
    } else {
        Err("R04 violation (#777): No save until there is a commit receipt.")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    macro_rules! rule_test {
        ($name:ident, $id:literal, $sentence:literal) => {
            #[test]
            fn $name() {
                let text = default_rulebook();
                assert!(text.contains(concat!("## ", $id)), "{}", $id);
                assert!(
                    text.contains($sentence),
                    "{} missing sentence: {}",
                    $id,
                    $sentence
                );
            }
        };
    }

    rule_test!(
        rule_r01_authority,
        "R01",
        "Recognized destructive calls are stopped before they run."
    );
    rule_test!(
        rule_r02_receipts,
        "R02",
        "Every stop, permit and allow gets a receipt."
    );
    rule_test!(
        rule_r03_provenance,
        "R03",
        "Every memory needs a causal anchor, or it is quarantined, never deleted."
    );
    rule_test!(rule_r04_commit, "R04", "No save until there is a commit.");
    rule_test!(
        rule_r05_shared_learning,
        "R05",
        "Every stop is saved to memory every agent shares."
    );
    rule_test!(
        rule_r06_truth,
        "R06",
        "A final message cannot cite a commit, file, PR or memory id that does not exist."
    );
    rule_test!(
        rule_r07_precedence,
        "R07",
        "The owner's rules beat agent-written rules"
    );
    rule_test!(
        rule_r08_session_start,
        "R08",
        "On startup, resume, clear, compact or fork, the agent loads this rulebook."
    );
    rule_test!(
        rule_r09_local,
        "R09",
        "Everything stays on the owner's machines"
    );
    rule_test!(
        rule_r10_witness,
        "R10",
        "No commit without a bisect witness after a failure."
    );
    rule_test!(rule_r11_loops, "R11", "Hard stop on repeat failures.");
    rule_test!(
        rule_r12_worktrees,
        "R12",
        "Refuse destructive worktree cleanup"
    );
    rule_test!(rule_r13_scope, "R13", "Contract before edits.");
    rule_test!(
        rule_r14_skeptical,
        "R14",
        "Memory is a claim, not live ground truth."
    );
    rule_test!(
        rule_r15_compaction,
        "R15",
        "Re-hydrate on every compaction, not just boot"
    );
    rule_test!(
        rule_r16_hostile,
        "R16",
        "Zero-trust git and repo ingestion."
    );
    rule_test!(
        rule_r17_upgrades,
        "R17",
        "Non-destructive migration with a backup receipt."
    );

    #[test]
    fn seventeen_sections_and_every_bullet_is_marked() {
        assert_eq!(section_ids().len(), 17);
        assert_eq!(VERSION, 1);
        assert!(TEXT.contains("version: 1"));
        for line in TEXT.lines().filter(|line| line.starts_with("- ")) {
            assert!(
                line.contains("SHIPPED") || line.contains("PARTIAL") || line.contains("DEFERRED"),
                "unmarked rule: {line}"
            );
        }
    }

    #[test]
    fn deferred_lines_name_an_issue() {
        for line in TEXT.lines() {
            if line.contains("DEFERRED") {
                assert!(
                    line.contains("DEFERRED:#"),
                    "a deferred rule names its issue: {line}"
                );
            }
        }
    }

    #[test]
    fn owner_rules_override_one_section_and_leave_the_rest() {
        let owner = "R07\nOwner text replaces the default precedence section.\n";
        let out = apply_owner(TEXT, [owner]);
        assert!(out.contains("Owner text replaces the default precedence section."));
        assert!(!out.contains("The owner's rules beat agent-written rules for the gate"));
        assert!(out.contains("## R01 Authority"));
        assert!(out.contains("## R08 Session start"));
        assert!(out.contains("Recognized destructive calls are stopped before they run."));
        let unknown = apply_owner(TEXT, ["R99\nno such section\n"]);
        assert_eq!(unknown, TEXT);
    }

    #[test]
    fn inject_cuts_rulebook_sections_before_the_pinned_rules() {
        let rules = "Vestige rules for this session (source: compact).\n- mem-7 [decision, tag pinned] Never force-push main.";
        let flags = vec!["CLAUDE.md:3 command `make` failed, receipt `mem-9`".to_string()];
        let text = inject(&[], Some(rules), &flags, 4096);
        assert!(text.len() <= 4096, "{}", text.len());
        assert!(text.contains(HANDLE), "{text}");
        assert!(
            text.contains("- mem-7 [decision, tag pinned] Never force-push main."),
            "{text}"
        );
        assert!(text.contains("Rules file check:"), "{text}");
        assert!(text.contains(&format!("omitted: `{HANDLE}`")), "{text}");
        assert!(text.contains("## R01 Authority"), "{text}");
        assert!(!text.contains("## R17"), "{text}");
        // With room for everything, nothing is cut.
        let full = inject(&[], Some(rules), &flags, 100_000);
        assert!(full.contains("## R17 Safe upgrades"), "{full}");
        assert!(!full.contains("omitted:"), "{full}");
        assert!(full.ends_with(&flags[0]), "{full}");
    }

    #[test]
    fn inject_keeps_the_handle_when_the_budget_is_small() {
        let text = inject(&[], None, &[], 80);
        assert!(text.contains(HANDLE), "{text}");
        assert!(text.len() <= 80, "{}", text.len());
    }

    #[test]
    fn cycle_f_loop_breaker_corrections_done_check_postmortem_and_elle_work() {
        // #542 & #549: Loop breaker stops on unchanged tree hash and surfaces dead ends + fix
        let attempts = vec![
            FailureAttempt {
                signature: "E0308".into(),
                tree_hash: "tree-a".into(),
                attempt_summary: "cast to u32".into(),
                receipt_id: "r1".into(),
                resolved_fix: None,
            },
            FailureAttempt {
                signature: "E0308".into(),
                tree_hash: "tree-a".into(),
                attempt_summary: "cast to usize".into(),
                receipt_id: "r2".into(),
                resolved_fix: None,
            },
        ];
        let dec = check_failure_loop("E0308", "tree-a", &attempts, 2);
        assert!(dec.blocked);
        assert_eq!(dec.dead_ends.len(), 2);
        let after_edit = check_failure_loop("E0308", "tree-b", &attempts, 2);
        assert!(!after_edit.blocked);

        // #544: Point-of-action corrections match command or path and stay quiet otherwise
        let corrs = vec![CorrectionAnchor {
            memory_id: "mem-42".into(),
            anchor_target: "git push --force".into(),
            rule_text: "Never force push".into(),
        }];
        assert_eq!(
            match_action_corrections("Bash", Some("git push --force origin"), None, &corrs).len(),
            1
        );
        assert!(match_action_corrections("Bash", Some("cargo test"), None, &corrs).is_empty());

        // #547: Done check flags fabricated SHA and missing mem-id
        let bad = verify_done_citations(
            "Landed in commit deadbeef1234 and saved mem-99999999.",
            &["70851594abcd"],
            &["mem-0000166a"],
            None,
        );
        assert_eq!(bad, vec!["deadbeef1234".to_string(), "mem-99999999".to_string()]);
        let good = verify_done_citations(
            "Landed in commit 70851594 and saved mem-0000166a.",
            &["70851594abcd"],
            &["mem-0000166a"],
            None,
        );
        assert!(good.is_empty());

        // #548: Postmortem pack orders events and rejects tampered entry
        let mut pack = build_postmortem_pack(&[
            (200, "stop", "blocked rm", "eff-2"),
            (100, "call", "read file", "eff-1"),
        ]);
        assert_eq!(pack[0].timestamp_ms, 100);
        assert!(verify_postmortem_pack(&pack).is_ok());
        pack[1].summary = "edited".into();
        assert_eq!(verify_postmortem_pack(&pack), Err(1));

        // #555: Elle lost update anomaly detection
        let writes = vec![
            AgentFileWrite {
                agent_id: "a1".into(),
                path: "src/config.rs".into(),
                read_hash: "h0".into(),
                write_hash: "h1".into(),
                receipt_id: "eff-a".into(),
            },
            AgentFileWrite {
                agent_id: "a2".into(),
                path: "src/config.rs".into(),
                read_hash: "h0".into(),
                write_hash: "h2".into(),
                receipt_id: "eff-b".into(),
            },
        ];
        let lost = detect_elle_lost_updates(&writes);
        assert_eq!(lost.len(), 1);
        assert_eq!(lost[0].1, "eff-a");
        assert_eq!(lost[0].2, "eff-b");

        // #777: Save-after-commit gate
        assert!(enforce_save_after_commit(true).is_ok());
        assert!(enforce_save_after_commit(false).is_err());
    }
}
