//! The verdict card: what was established about the first bad commit, one
//! rung at a time, each with whether it holds and the runs that back it.
//!
//! * LEAD: the walk had reached the commit over recorded links.
//! * BOUNDARY: the commit fails the test and its parent passes.
//! * CONFIRMED: stock `git bisect` over the whole range names it.
//! * ISOLATED: a part of the commit alone causes the failure and the rest
//!   passes without it (or the commit is a single change).
//! * REVERSED: undoing it on the bad ref makes the test pass again.
//! * REPEATED (`--flaky`): it fails more often with the commit than
//!   without, beyond chance.
//!
//! A rung's statement follows what happened: a rung that does not hold says
//! what was found instead, never the claim it failed to make.

use serde_json::{Value, json};

use super::hunks::Unit;
use super::json::Entry;
use super::probe::text_of;
use super::stats::{REPEATED_P, fisher_p, rate_lower, rate_upper};
use super::text::{format_g, py_display};

/// What the search inside the first bad commit found.
#[derive(Debug, Default)]
pub(super) struct Why {
    /// Verdict of the undo on the bad ref, or `rewritten` when nothing applied.
    pub revert: Option<String>,
    pub undo_how: Option<&'static str>,
    /// Verdict of the commit without the minimal set, or `does not apply`.
    pub without: Option<String>,
    /// How many independently applicable changes the commit makes.
    pub units: usize,
    /// The smallest set of them found to fail on the parent.
    pub minimal: Vec<Unit>,
    /// Whether the search ended on its own, not on the run budget.
    pub complete: bool,
    /// The undo that was tested, as a patch on the bad ref.
    pub undo_patch: Option<Vec<u8>>,
    /// The commit has no parent, so its changes were not searched.
    pub root: bool,
}

/// One side of the fixed-size strength runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct Side {
    pub fails: u64,
    pub runs: u64,
    /// The number of the probe that holds these runs.
    pub probe: u64,
}

/// `--flaky`: how much more often the test fails with the first bad commit
/// than without it.
#[derive(Debug, Clone, PartialEq)]
pub(super) struct Strength {
    pub with: Side,
    pub without: Side,
    /// One-sided Fisher exact p of that split.
    pub fisher_p: f64,
    /// Exact 97.5% lower bound on the failure rate with the commit.
    pub rate_with_at_least: f64,
    /// Exact 97.5% upper bound on the failure rate without it.
    pub rate_without_at_most: f64,
    /// The first bound over the second.
    pub at_least_times: Option<f64>,
}

impl Strength {
    pub fn measured(with: Side, without: Side) -> Self {
        let lower = rate_lower(with.fails, with.runs);
        let upper = rate_upper(without.fails, without.runs);
        Self {
            with,
            without,
            fisher_p: fisher_p(with.fails, with.runs, without.fails, without.runs),
            rate_with_at_least: lower,
            rate_without_at_most: upper,
            at_least_times: (upper > 0.0).then(|| lower / upper),
        }
    }

    pub fn json(&self) -> Value {
        let side =
            |side: &Side| json!({"fails": side.fails, "runs": side.runs, "probe": side.probe});
        json!({
            "with": side(&self.with),
            "without": side(&self.without),
            "fisher_p": self.fisher_p,
            "rate_with_at_least": self.rate_with_at_least,
            "rate_without_at_most": self.rate_without_at_most,
            "at_least_times": self.at_least_times,
        })
    }
}

/// One rung of the card.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct Rung {
    pub name: &'static str,
    pub holds: bool,
    pub statement: String,
    /// The runs, or the recorded link, the rung rests on.
    pub proof: String,
}

impl Rung {
    pub fn json(&self) -> Value {
        json!({
            "rung": self.name,
            "holds": self.holds,
            "statement": self.statement,
            "proof": self.proof,
        })
    }
}

/// Everything the card is read from.
pub(super) struct CardFacts<'a> {
    pub entries: &'a [Entry],
    pub first_bad: &'a str,
    /// The first parent of the first bad commit; `None` for a root commit.
    pub parent: Option<&'a str>,
    /// The lead that named the commit: its rank, the number of leads, and
    /// the memory that is the recorded link.
    pub lead: Option<(usize, usize, &'a str)>,
    /// Commits in `good..bad`.
    pub window: u64,
    /// How many recorded verdicts contradict this being the first bad
    /// commit: bad on a commit before it, or good on one that contains it.
    pub contradictions: usize,
    pub bad_ref: &'a str,
    pub why: Option<&'a Why>,
    pub strength: Option<&'a Strength>,
}

/// `runs 6,7`, or `no run`.
fn runs(backing: &[&Entry]) -> String {
    if backing.is_empty() {
        return "no run".to_string();
    }
    let numbers: Vec<String> = backing
        .iter()
        .map(|entry| py_display(entry.get("n")))
        .collect();
    format!("runs {}", numbers.join(","))
}

fn isolated(why: &Why, boundary: (bool, &str), searched: String) -> Rung {
    let count = why.minimal.len();
    let total = why.units;
    let it = if count == 1 { "it" } else { "them" };
    let found = format!("{count} of its {total} changes alone causes it");
    let (holds, statement, proof) = if why.root {
        (
            false,
            "a root commit has no parent to apply its changes to, so they were not searched"
                .to_string(),
            searched,
        )
    } else if total == 0 {
        (
            false,
            "the commit changes nothing against its parent, so there was nothing to search"
                .to_string(),
            searched,
        )
    } else if total == 1 {
        // Nothing to take away from one change: what isolates it is that
        // the commit fails and the commit before it passes, so the rung
        // holds exactly when those two runs say so.
        (
            boundary.0,
            "the commit is one change, and the commit before it passes".to_string(),
            boundary.1.to_string(),
        )
    } else if count < total {
        match why.without.as_deref() {
            Some("good") => (
                true,
                format!("{found}, and the rest passes without {it}"),
                searched,
            ),
            _ => (
                false,
                format!("{found}, but the rest was not shown to pass without {it}"),
                searched,
            ),
        }
    } else if why.complete {
        (
            false,
            format!("no smaller part of its {total} changes causes it alone"),
            searched,
        )
    } else {
        (
            false,
            format!("its {total} changes were not narrowed down before the run budget ran out"),
            searched,
        )
    };
    Rung {
        name: "ISOLATED",
        holds,
        statement,
        proof,
    }
}

fn reversed(why: &Why, bad_ref: &str, undone: String) -> Rung {
    let what = if why.undo_how == Some("whole commit") {
        "the whole commit"
    } else {
        "just those lines"
    };
    let (holds, statement, proof) = match why.revert.as_deref() {
        Some("good") => (
            true,
            format!("undoing {what} on {bad_ref} makes the test pass again"),
            undone,
        ),
        Some("bad") => (
            false,
            format!(
                "undoing {what} on {bad_ref} does not make the test pass; something later also carries it"
            ),
            undone,
        ),
        Some("skip") => (
            false,
            format!("undoing {what} on {bad_ref} could not be tested"),
            undone,
        ),
        _ => (
            false,
            format!("later commits rewrote these lines on {bad_ref}; there is no mechanical undo"),
            "no run".to_string(),
        ),
    };
    Rung {
        name: "REVERSED",
        holds,
        statement,
        proof,
    }
}

/// The card for a first bad commit.
pub(super) fn verdict_card(facts: &CardFacts<'_>) -> Vec<Rung> {
    let in_phase = |phase: &str| -> Vec<&Entry> {
        facts
            .entries
            .iter()
            .filter(|entry| text_of(entry, "phase") == phase)
            .collect()
    };
    let first_on = |commit: &str| -> Vec<&Entry> {
        facts
            .entries
            .iter()
            .filter(|entry| text_of(entry, "commit") == commit)
            .take(1)
            .collect()
    };
    let recorded = |on: &[&Entry], verdict: &str| {
        on.first()
            .is_some_and(|entry| text_of(entry, "verdict") == verdict)
    };

    let mut card = Vec::new();
    card.push(match facts.lead {
        Some((rank, leads, memory)) => Rung {
            name: "LEAD",
            holds: true,
            statement: format!("the walk reached it over recorded links (lead {rank} of {leads})"),
            proof: format!("recorded link {memory}"),
        },
        None => Rung {
            name: "LEAD",
            holds: false,
            statement: "the walk reached it over recorded links".to_string(),
            proof: "the walk did not reach it".to_string(),
        },
    });

    let on_first_bad = first_on(facts.first_bad);
    let on_parent = facts.parent.map(first_on).unwrap_or_default();
    let boundary_holds = recorded(&on_first_bad, "bad") && recorded(&on_parent, "good");
    let both: Vec<&Entry> = on_first_bad.iter().chain(&on_parent).copied().collect();
    let boundary_runs = runs(&both);
    card.push(Rung {
        name: "BOUNDARY",
        holds: boundary_holds,
        statement: "it fails the test and the commit before it passes".to_string(),
        proof: boundary_runs.clone(),
    });

    let bisected = in_phase("bisect");
    let named = format!(
        "stock git bisect over all {} commits names it",
        facts.window
    );
    let bisect_runs = if bisected.is_empty() {
        "reused runs".to_string()
    } else {
        runs(&bisected)
    };
    card.push(if facts.contradictions == 0 {
        Rung {
            name: "CONFIRMED",
            holds: true,
            statement: named,
            proof: bisect_runs,
        }
    } else {
        // git bisect names a commit whatever the other runs say; the rung
        // only holds when none of them says otherwise.
        Rung {
            name: "CONFIRMED",
            holds: false,
            statement: format!(
                "git bisect names it, but {} recorded verdict{} contradict it",
                facts.contradictions,
                if facts.contradictions == 1 { "" } else { "s" }
            ),
            proof: "see the runs listed above".to_string(),
        }
    });

    if let Some(why) = facts.why {
        let searched: Vec<&Entry> = in_phase("lines")
            .into_iter()
            .chain(in_phase("without"))
            .collect();
        card.push(isolated(
            why,
            (boundary_holds, &boundary_runs),
            runs(&searched),
        ));
        card.push(reversed(why, facts.bad_ref, runs(&in_phase("undo"))));
    }

    if let Some(strength) = facts.strength {
        card.push(Rung {
            name: "REPEATED",
            holds: strength.fisher_p < REPEATED_P,
            statement: format!(
                "it fails {} of {} times with the commit and {} of {} without (p = {})",
                strength.with.fails,
                strength.with.runs,
                strength.without.fails,
                strength.without.runs,
                format_g(strength.fisher_p, 2)
            ),
            proof: format!("runs {},{}", strength.with.probe, strength.without.probe),
        });
    }
    card
}

#[cfg(test)]
mod tests {
    use super::*;

    const FIRST_BAD: &str = "5555555555555555555555555555555555555555";
    const PARENT: &str = "4444444444444444444444444444444444444444";

    fn probe(n: u64, commit: &str, verdict: &str, phase: &str) -> Entry {
        json!({"n": n, "commit": commit, "verdict": verdict, "phase": phase})
            .as_object()
            .cloned()
            .unwrap()
    }

    /// A run like the planted regression: the commit is a lead, it and its
    /// parent were probed in step 4, bisect ran two more, then the search.
    fn entries() -> Vec<Entry> {
        vec![
            probe(1, "1111", "good", "baseline"),
            probe(2, "9999", "bad", "baseline"),
            probe(3, FIRST_BAD, "bad", "candidate"),
            probe(4, "2222", "good", "candidate"),
            probe(5, PARENT, "good", "parent"),
            probe(6, "7777", "bad", "bisect"),
            probe(7, "3333", "good", "bisect"),
            probe(8, "4444444444 + 2 of 4 changes", "bad", "lines"),
            probe(9, "4444444444 + 1 of 4 changes", "bad", "lines"),
            probe(10, "4444444444 + the other 3 changes", "good", "without"),
            probe(11, "9999999999 with 5555555555 undone", "good", "undo"),
        ]
    }

    fn unit(file: &str) -> Unit {
        Unit {
            file: file.to_string(),
            header: Vec::new(),
            body: Vec::new(),
            start: 1,
            added: Vec::new(),
        }
    }

    fn why(units: usize, minimal: usize, without: Option<&str>, revert: &str) -> Why {
        Why {
            revert: Some(revert.to_string()),
            undo_how: Some("whole commit"),
            without: without.map(str::to_string),
            units,
            minimal: (0..minimal).map(|i| unit(&format!("f{i}"))).collect(),
            complete: true,
            undo_patch: None,
            root: false,
        }
    }

    fn facts<'a>(entries: &'a [Entry], why: Option<&'a Why>) -> CardFacts<'a> {
        CardFacts {
            entries,
            first_bad: FIRST_BAD,
            parent: Some(PARENT),
            lead: Some((3, 7, "mem-00000000000000dd")),
            window: 12,
            contradictions: 0,
            bad_ref: "v2",
            why,
            strength: None,
        }
    }

    fn row(rung: &Rung) -> (&str, bool, &str, &str) {
        (rung.name, rung.holds, &rung.statement, &rung.proof)
    }

    #[test]
    fn five_rungs_hold_for_a_planted_regression() {
        let entries = entries();
        let why = why(4, 1, Some("good"), "good");
        let card = verdict_card(&facts(&entries, Some(&why)));
        let rows: Vec<_> = card.iter().map(row).collect();
        assert_eq!(
            rows,
            [
                (
                    "LEAD",
                    true,
                    "the walk reached it over recorded links (lead 3 of 7)",
                    "recorded link mem-00000000000000dd"
                ),
                (
                    "BOUNDARY",
                    true,
                    "it fails the test and the commit before it passes",
                    "runs 3,5"
                ),
                (
                    "CONFIRMED",
                    true,
                    "stock git bisect over all 12 commits names it",
                    "runs 6,7"
                ),
                (
                    "ISOLATED",
                    true,
                    "1 of its 4 changes alone causes it, and the rest passes without it",
                    "runs 8,9,10"
                ),
                (
                    "REVERSED",
                    true,
                    "undoing the whole commit on v2 makes the test pass again",
                    "runs 11"
                ),
            ]
        );
        assert_eq!(
            card[1].json(),
            json!({
                "rung": "BOUNDARY",
                "holds": true,
                "statement": "it fails the test and the commit before it passes",
                "proof": "runs 3,5",
            })
        );
    }

    #[test]
    fn without_the_line_search_the_card_has_three_rungs() {
        let entries = entries();
        let card = verdict_card(&facts(&entries, None));
        let names: Vec<_> = card.iter().map(|rung| rung.name).collect();
        assert_eq!(names, ["LEAD", "BOUNDARY", "CONFIRMED"]);
    }

    #[test]
    fn a_commit_the_walk_missed_has_no_lead() {
        let entries = entries();
        let mut facts = facts(&entries, None);
        facts.lead = None;
        let card = verdict_card(&facts);
        assert_eq!(
            row(&card[0]),
            (
                "LEAD",
                false,
                "the walk reached it over recorded links",
                "the walk did not reach it"
            )
        );
        // The tested rungs do not depend on the walk.
        assert!(card[1].holds && card[2].holds);
    }

    #[test]
    fn the_boundary_needs_a_run_on_each_side() {
        // The parent was never tested itself (a good descendant covers it).
        let mut entries = entries();
        entries.retain(|entry| text_of(entry, "commit") != PARENT);
        let card = verdict_card(&facts(&entries, None));
        assert_eq!((card[1].holds, card[1].proof.as_str()), (false, "runs 3"));
        // A root commit has no parent at all.
        let mut rooted = facts(&entries, None);
        rooted.parent = None;
        assert!(!verdict_card(&rooted)[1].holds);
        // A parent that fails too is no boundary.
        let mut entries = self::entries();
        entries[4] = probe(5, PARENT, "bad", "parent");
        let card = verdict_card(&facts(&entries, None));
        assert_eq!((card[1].holds, card[1].proof.as_str()), (false, "runs 3,5"));
        // Every verdict reused from step 4: bisect ran nothing new.
        let mut entries = self::entries();
        entries.retain(|entry| text_of(entry, "phase") != "bisect");
        let card = verdict_card(&facts(&entries, None));
        assert_eq!(
            (card[2].holds, card[2].proof.as_str()),
            (true, "reused runs")
        );
    }

    #[test]
    fn confirmed_holds_only_when_no_recorded_run_says_otherwise() {
        let entries = entries();
        let mut facts = facts(&entries, None);
        facts.contradictions = 1;
        let card = verdict_card(&facts);
        assert_eq!(
            row(&card[2]),
            (
                "CONFIRMED",
                false,
                "git bisect names it, but 1 recorded verdict contradict it",
                "see the runs listed above"
            )
        );
        facts.contradictions = 2;
        assert_eq!(
            verdict_card(&facts)[2].statement,
            "git bisect names it, but 2 recorded verdicts contradict it"
        );
        // The other rungs are read from their own runs.
        assert!(card[0].holds && card[1].holds);
    }

    #[test]
    fn isolated_says_what_the_search_found() {
        let entries = entries();
        let isolated = |why: &Why| {
            let card = verdict_card(&facts(&entries, Some(why)));
            (
                card[3].holds,
                card[3].statement.clone(),
                card[3].proof.clone(),
            )
        };
        // A single change: nothing to take away, the boundary is the proof.
        assert_eq!(
            isolated(&why(1, 1, None, "good")),
            (
                true,
                "the commit is one change, and the commit before it passes".to_string(),
                "runs 3,5".to_string()
            )
        );
        assert_eq!(
            isolated(&why(4, 2, Some("good"), "good")).1,
            "2 of its 4 changes alone causes it, and the rest passes without them"
        );
        // The rest was tested and fails, does not apply, or could not be
        // tested: either way it was not shown to pass.
        for without in ["bad", "does not apply", "skip"] {
            assert_eq!(
                isolated(&why(4, 1, Some(without), "good")),
                (
                    false,
                    "1 of its 4 changes alone causes it, but the rest was not shown to pass without it"
                        .to_string(),
                    "runs 8,9,10".to_string()
                ),
                "{without}"
            );
        }
        assert_eq!(
            isolated(&why(4, 3, Some("bad"), "good")).1,
            "3 of its 4 changes alone causes it, but the rest was not shown to pass without them"
        );
        // Every change is needed: there is no rest to test.
        assert_eq!(
            isolated(&why(4, 4, None, "good")),
            (
                false,
                "no smaller part of its 4 changes causes it alone".to_string(),
                "runs 8,9,10".to_string()
            )
        );
        let mut unfinished = why(4, 4, None, "good");
        unfinished.complete = false;
        assert_eq!(
            isolated(&unfinished).1,
            "its 4 changes were not narrowed down before the run budget ran out"
        );
        let mut root = why(0, 0, None, "good");
        root.root = true;
        assert_eq!(
            isolated(&root).1,
            "a root commit has no parent to apply its changes to, so they were not searched"
        );
        assert!(!isolated(&why(0, 0, None, "good")).0);
    }

    #[test]
    fn a_single_change_is_isolated_only_with_its_boundary() {
        let mut entries = entries();
        entries.retain(|entry| text_of(entry, "commit") != PARENT);
        let why = why(1, 1, None, "good");
        let card = verdict_card(&facts(&entries, Some(&why)));
        assert_eq!(
            row(&card[3]),
            (
                "ISOLATED",
                false,
                "the commit is one change, and the commit before it passes",
                "runs 3"
            )
        );
    }

    #[test]
    fn reversed_says_what_the_undo_did() {
        let entries = entries();
        let reversed = |why: &Why| {
            let card = verdict_card(&facts(&entries, Some(why)));
            (card[4].holds, card[4].statement.clone())
        };
        let mut lines_only = why(4, 1, Some("good"), "good");
        lines_only.undo_how = Some("found lines only");
        assert_eq!(
            reversed(&lines_only),
            (
                true,
                "undoing just those lines on v2 makes the test pass again".to_string()
            )
        );
        assert_eq!(
            reversed(&why(4, 1, Some("good"), "bad")),
            (
                false,
                "undoing the whole commit on v2 does not make the test pass; something later also carries it".to_string()
            )
        );
        lines_only.revert = Some("bad".to_string());
        assert_eq!(
            reversed(&lines_only).1,
            "undoing just those lines on v2 does not make the test pass; something later also carries it"
        );
        assert_eq!(
            reversed(&why(4, 1, Some("good"), "skip")),
            (
                false,
                "undoing the whole commit on v2 could not be tested".to_string()
            )
        );
        // Nothing applied: no run stands behind the rung.
        let mut entries = self::entries();
        entries.retain(|entry| text_of(entry, "phase") != "undo");
        let mut rewritten = why(4, 1, Some("good"), "rewritten");
        rewritten.undo_how = None;
        let card = verdict_card(&facts(&entries, Some(&rewritten)));
        assert_eq!(
            row(&card[4]),
            (
                "REVERSED",
                false,
                "later commits rewrote these lines on v2; there is no mechanical undo",
                "no run"
            )
        );
    }

    #[test]
    fn repeated_holds_when_the_split_is_beyond_chance() {
        let entries = entries();
        let with = Side {
            fails: 15,
            runs: 50,
            probe: 12,
        };
        let without = Side {
            fails: 0,
            runs: 50,
            probe: 13,
        };
        let strength = Strength::measured(with, without);
        // The reference tool's figures for 15 of 50 against 0 of 50.
        assert!((strength.fisher_p - 8.884673390211095e-06).abs() < 1e-17);
        assert!((strength.rate_with_at_least - 0.1786178456641469).abs() < 1e-13);
        assert!((strength.rate_without_at_most - 0.07112173646419767).abs() < 1e-13);
        assert!((strength.at_least_times.unwrap() - 2.511438197998192).abs() < 1e-11);
        let mut facts = facts(&entries, None);
        facts.strength = Some(&strength);
        let card = verdict_card(&facts);
        assert_eq!(
            row(&card[3]),
            (
                "REPEATED",
                true,
                "it fails 15 of 50 times with the commit and 0 of 50 without (p = 8.9e-06)",
                "runs 12,13"
            )
        );
        assert_eq!(
            strength.json()["with"],
            json!({"fails": 15, "runs": 50, "probe": 12})
        );
        assert_eq!(strength.json()["fisher_p"], json!(strength.fisher_p));

        // Too few failures to tell apart from chance.
        let weak = Strength::measured(Side { fails: 3, ..with }, without);
        facts.strength = Some(&weak);
        let card = verdict_card(&facts);
        assert!(!card[3].holds);
        assert!(
            card[3].statement.ends_with("(p = 0.12)"),
            "{}",
            card[3].statement
        );

        // Nothing could be run on either side: no figure is invented.
        let none = Side {
            fails: 0,
            runs: 0,
            probe: 12,
        };
        let empty = Strength::measured(none, none);
        assert_eq!(empty.fisher_p, 1.0);
        assert_eq!(empty.at_least_times, Some(0.0));
        // Fails every time without the commit too: no finite ratio.
        let always = Side {
            fails: 50,
            runs: 50,
            probe: 13,
        };
        let same = Strength::measured(always, always);
        assert_eq!(same.rate_without_at_most, 1.0);
        assert!(same.at_least_times.unwrap() < 1.0);
        assert!(same.fisher_p >= REPEATED_P);
    }
}
