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
/// pinned-rule block, then rules-file flags. Whole sections drop from the
/// end to stay inside `max_bytes`. The version handle is never cut.
pub fn inject(owner: &[&str], rules: Option<&str>, flags: &[String], max_bytes: usize) -> String {
    let book = apply_owner(TEXT, owner.iter().copied());
    let mut parts: Vec<String> = vec![book];
    if let Some(rules) = rules.filter(|text| !text.is_empty()) {
        parts.push(rules.to_string());
    }
    if !flags.is_empty() {
        let mut block = String::from("Rules file check:");
        for flag in flags {
            block.push('\n');
            block.push_str(flag);
        }
        parts.push(block);
    }
    let mut text = parts.join("\n\n");
    if text.len() <= max_bytes {
        return text;
    }
    // Drop trailing sections of the rulebook, then the later parts, keeping
    // the version handle on its own line when nothing else fits.
    while text.len() > max_bytes {
        if let Some(idx) = text.rfind("\n## R") {
            text.truncate(idx);
            text.push_str("\nomitted: `");
            text.push_str(HANDLE);
            text.push('`');
            if text.len() <= max_bytes {
                break;
            }
            // The omitted line itself must fit; drop it and truncate further.
            if let Some(omit) = text.rfind("\nomitted:") {
                text.truncate(omit);
            }
            continue;
        }
        if let Some(idx) = text.rfind("\n\n") {
            text.truncate(idx);
            continue;
        }
        text = format!("handle: `{HANDLE}`");
        break;
    }
    if text.len() > max_bytes {
        text = format!("handle: `{HANDLE}`");
    }
    text
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
    fn inject_keeps_the_handle_when_the_budget_is_small() {
        let text = inject(&[], None, &[], 80);
        assert!(text.contains(HANDLE), "{text}");
        assert!(text.len() <= 80, "{}", text.len());
    }
}
