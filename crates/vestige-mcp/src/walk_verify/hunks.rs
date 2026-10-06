//! A commit's changes as independently applicable units, and Zeller's ddmin
//! over them.

use std::cell::Cell;

use super::text::split_lines;

/// One independently applicable change of a commit: a hunk, or the whole
/// file diff for a file that is added, deleted, or has no hunks (binary,
/// mode change, pure rename).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Unit {
    pub file: String,
    /// The file's diff header, repeated before the hunks of one file. Empty
    /// for a whole-file unit, whose `body` carries its own header.
    pub header: Vec<u8>,
    pub body: Vec<u8>,
    /// First line of the hunk in the new file; 0 for a whole-file unit.
    pub start: u64,
    /// The lines the hunk adds, for display.
    pub added: Vec<String>,
}

/// Offsets of the lines of `text` that start with `prefix`.
fn line_starts_with(text: &[u8], prefix: &[u8]) -> Vec<usize> {
    let mut offsets = Vec::new();
    let mut start = 0;
    while start < text.len() {
        if text[start..].starts_with(prefix) {
            offsets.push(start);
        }
        match text[start..].iter().position(|byte| *byte == b'\n') {
            Some(newline) => start += newline + 1,
            None => break,
        }
    }
    offsets
}

/// The line that begins at `offset`, without its newline.
fn line_at(text: &[u8], offset: usize) -> &[u8] {
    let rest = text.get(offset..).unwrap_or_default();
    let end = rest
        .iter()
        .position(|byte| *byte == b'\n')
        .unwrap_or(rest.len());
    &rest[..end]
}

fn contains(haystack: &[u8], needle: &[u8]) -> bool {
    !needle.is_empty()
        && haystack
            .windows(needle.len())
            .any(|window| window == needle)
}

/// The path a file diff names in its header: `+++ b/<path>`, else
/// `--- a/<path>`, else the header's first line. git ends a name that holds
/// a space with a tab; that tab is not part of the name.
fn diff_path(header: &[u8]) -> String {
    for prefix in [&b"+++ b/"[..], &b"--- a/"[..]] {
        for offset in line_starts_with(header, prefix) {
            let rest = &line_at(header, offset)[prefix.len()..];
            let rest = rest.strip_suffix(b"\t").unwrap_or(rest);
            if !rest.is_empty() {
                return String::from_utf8_lossy(rest).into_owned();
            }
        }
    }
    String::from_utf8_lossy(line_at(header, 0)).into_owned()
}

/// `N` of a hunk header `@@ -a[,b] +N[,d] @@`; 0 when it does not parse.
fn hunk_start(hunk: &[u8]) -> u64 {
    let line = String::from_utf8_lossy(line_at(hunk, 0)).into_owned();
    let parse = || -> Option<u64> {
        let rest = line.strip_prefix("@@ -")?;
        let rest = rest.trim_start_matches(|c: char| c.is_ascii_digit());
        let rest = match rest.strip_prefix(',') {
            Some(count) => count.trim_start_matches(|c: char| c.is_ascii_digit()),
            None => rest,
        };
        let rest = rest.strip_prefix(" +")?;
        let digits: String = rest.chars().take_while(char::is_ascii_digit).collect();
        digits.parse().ok()
    };
    parse().unwrap_or(0)
}

/// Split a unified diff into units: one per hunk, or the whole file diff for
/// new, deleted and hunkless files.
pub fn split_hunks(diff: &[u8]) -> Vec<Unit> {
    let mut units = Vec::new();
    let files = line_starts_with(diff, b"diff --git ");
    for (index, &begin) in files.iter().enumerate() {
        let end = files.get(index + 1).copied().unwrap_or(diff.len());
        let file_diff = &diff[begin..end];
        let hunks = line_starts_with(file_diff, b"@@ ");
        let header = &file_diff[..hunks.first().copied().unwrap_or(file_diff.len())];
        let path = diff_path(header);
        let whole = hunks.is_empty()
            || contains(header, b"new file mode")
            || contains(header, b"deleted file mode");
        if whole {
            units.push(Unit {
                file: path,
                header: Vec::new(),
                body: file_diff.to_vec(),
                start: 0,
                added: Vec::new(),
            });
            continue;
        }
        for (position, &hunk_begin) in hunks.iter().enumerate() {
            let hunk_end = hunks.get(position + 1).copied().unwrap_or(file_diff.len());
            let hunk = &file_diff[hunk_begin..hunk_end];
            let text = String::from_utf8_lossy(hunk);
            let added = split_lines(&text)
                .into_iter()
                .skip(1)
                .filter_map(|line| line.strip_prefix('+'))
                .map(str::to_string)
                .collect();
            units.push(Unit {
                file: path.clone(),
                header: header.to_vec(),
                body: hunk.to_vec(),
                start: hunk_start(hunk),
                added,
            });
        }
    }
    units
}

/// The patch that applies exactly these units, in diff order: each file's
/// header once, then its hunks.
pub fn patch_of(units: &[&Unit]) -> Vec<u8> {
    let mut patch = Vec::new();
    let mut last: Option<&[u8]> = None;
    for unit in units {
        if unit.header.is_empty() || last != Some(unit.header.as_slice()) {
            patch.extend_from_slice(&unit.header);
            last = Some(unit.header.as_slice());
        }
        patch.extend_from_slice(&unit.body);
    }
    patch
}

/// `file:line` of each unit, comma separated.
pub(super) fn names_of(units: &[&Unit]) -> String {
    units
        .iter()
        .map(|unit| format!("{}:{}", unit.file, unit.start))
        .collect::<Vec<_>>()
        .join(", ")
}

/// Zeller's ddmin: shrink `items` to a 1-minimal subset for which `fails`
/// still holds. `fails` spends the budget; the search stops when it is used
/// up, and the subset returned then still fails but may shrink further.
pub fn ddmin<T, F>(items: Vec<T>, budget: &Cell<i64>, mut fails: F) -> Vec<T>
where
    T: Clone + PartialEq,
    F: FnMut(&[T]) -> bool,
{
    let mut items = items;
    let mut n = 2usize;
    while items.len() >= 2 && budget.get() > 0 {
        let size = std::cmp::max(1, items.len() / n);
        let subsets: Vec<Vec<T>> = items.chunks(size).map(<[T]>::to_vec).collect();
        let mut moved = false;
        for subset in &subsets {
            if budget.get() <= 0 {
                break;
            }
            if fails(subset) {
                items = subset.clone();
                n = 2;
                moved = true;
                break;
            }
        }
        if !moved {
            for subset in &subsets {
                if budget.get() <= 0 || subsets.len() <= 2 {
                    break;
                }
                let complement: Vec<T> = items
                    .iter()
                    .filter(|item| !subset.contains(item))
                    .cloned()
                    .collect();
                if fails(&complement) {
                    items = complement;
                    n = std::cmp::max(n - 1, 2);
                    moved = true;
                    break;
                }
            }
        }
        if !moved {
            if n >= items.len() {
                break;
            }
            n = std::cmp::min(items.len(), n * 2);
        }
    }
    items
}

#[cfg(test)]
mod tests {
    use super::*;

    const DIFF: &str = r"diff --git a/src/calc.sh b/src/calc.sh
index 1111111..2222222 100644
--- a/src/calc.sh
+++ b/src/calc.sh
@@ -1,3 +1,4 @@
 #!/bin/sh
+# a comment
 a=1
 b=2
@@ -10,3 +11,3 @@ footer
 x
-echo $((a + b))
+echo $((a - b))
 y
\ No newline at end of file
diff --git a/new.txt b/new.txt
new file mode 100644
index 0000000..3333333
--- /dev/null
+++ b/new.txt
@@ -0,0 +1 @@
+hello
diff --git a/old.txt b/old.txt
deleted file mode 100644
index 4444444..0000000
--- a/old.txt
+++ /dev/null
@@ -1 +0,0 @@
-bye
diff --git a/logo.png b/logo.png
index 5555555..6666666 100644
Binary files a/logo.png and b/logo.png differ
";

    #[test]
    fn a_diff_splits_into_one_unit_per_hunk_and_whole_files() {
        let units = split_hunks(DIFF.as_bytes());
        let shape: Vec<(&str, u64, usize)> = units
            .iter()
            .map(|unit| (unit.file.as_str(), unit.start, unit.added.len()))
            .collect();
        assert_eq!(
            shape,
            [
                ("src/calc.sh", 1, 1),
                ("src/calc.sh", 11, 1),
                ("new.txt", 0, 0),
                ("old.txt", 0, 0),
                ("diff --git a/logo.png b/logo.png", 0, 0),
            ]
        );
        assert_eq!(units[0].added, ["# a comment"]);
        assert_eq!(units[1].added, ["echo $((a - b))"]);
        assert!(units[0].header.starts_with(b"diff --git a/src/calc.sh"));
        assert!(units[0].header.ends_with(b"+++ b/src/calc.sh\n"));
        assert_eq!(units[0].header, units[1].header);
        assert!(units[1].body.ends_with(b"\\ No newline at end of file\n"));
        // Whole-file units carry their own header.
        assert!(units[2].header.is_empty());
        assert!(units[2].body.starts_with(b"diff --git a/new.txt"));
        assert!(units[4].body.ends_with(b"differ\n"));
        assert!(split_hunks(b"").is_empty());
        assert!(split_hunks(b"not a diff\n").is_empty());
    }

    #[test]
    fn a_file_name_comes_from_the_header_only() {
        // git marks the end of a name that holds a space with a tab.
        let spaced = b"diff --git a/my file.txt b/my file.txt\nindex 1..2 100644\n--- a/my file.txt\t\n+++ b/my file.txt\t\n@@ -1 +1 @@\n-a\n+b\n";
        let units = split_hunks(spaced);
        assert_eq!(units.len(), 1);
        assert_eq!(units[0].file, "my file.txt");
        // A removed line that reads `-- a/other` shows as `--- a/other` in
        // the hunk; it is content, not the file's name.
        let tricky = b"diff --git a/gone.txt b/gone.txt\ndeleted file mode 100644\nindex 1..0\n--- a/gone.txt\n+++ /dev/null\n@@ -1,2 +0,0 @@\n--- a/other\n-x\n";
        let units = split_hunks(tricky);
        assert_eq!(units[0].file, "gone.txt");
        // A diff cut off in the middle of a header still splits.
        let cut = b"diff --git a/x b/x\n--- a/x";
        assert_eq!(split_hunks(cut)[0].file, "x");
        // A binary patch has no hunks: one whole-file unit.
        let binary = b"diff --git a/logo.png b/logo.png\nindex 5555555..6666666 100644\nGIT binary patch\nliteral 3\nKcmZQzU\n\nliteral 2\nJcmZQz\n\n";
        let units = split_hunks(binary);
        assert_eq!(units.len(), 1);
        assert_eq!(units[0].body, binary);
    }

    #[test]
    fn a_patch_of_units_repeats_each_file_header_once() {
        let units = split_hunks(DIFF.as_bytes());
        let all: Vec<&Unit> = units.iter().collect();
        assert_eq!(patch_of(&all), DIFF.as_bytes());
        // The second hunk alone still gets its file header.
        let second = patch_of(&[&units[1]]);
        let text = String::from_utf8(second).unwrap();
        assert!(text.starts_with("diff --git a/src/calc.sh b/src/calc.sh\n"));
        assert!(text.contains("@@ -10,3 +11,3 @@ footer\n"));
        assert!(!text.contains("# a comment"));
        assert_eq!(text.matches("+++ b/src/calc.sh").count(), 1);
        let both = String::from_utf8(patch_of(&[&units[0], &units[1]])).unwrap();
        assert_eq!(both.matches("+++ b/src/calc.sh").count(), 1);
        assert_eq!(
            names_of(&[&units[0], &units[1]]),
            "src/calc.sh:1, src/calc.sh:11"
        );
        assert!(patch_of(&[]).is_empty());
        assert_eq!(hunk_start(b"@@ -843 +843,8 @@ def connect"), 843);
        assert_eq!(hunk_start(b"@@ -10,3 +11,3 @@ footer"), 11);
        assert_eq!(hunk_start(b"not a hunk"), 0);
        assert_eq!(hunk_start(b""), 0);
    }

    /// `fails` for a failure that needs every item of `cause`; counts runs
    /// and spends the budget like the real test does.
    fn needs<'a>(
        cause: &'a [u32],
        budget: &'a Cell<i64>,
        runs: &'a Cell<u32>,
    ) -> impl FnMut(&[u32]) -> bool + 'a {
        move |subset| {
            budget.set(budget.get() - 1);
            runs.set(runs.get() + 1);
            cause.iter().all(|item| subset.contains(item))
        }
    }

    #[test]
    fn ddmin_finds_the_one_change_that_fails() {
        let (budget, runs) = (Cell::new(100), Cell::new(0));
        let found = ddmin((0..8).collect(), &budget, needs(&[5], &budget, &runs));
        assert_eq!(found, [5]);
        assert!(runs.get() <= 8, "{} runs", runs.get());
    }

    #[test]
    fn ddmin_keeps_changes_that_only_fail_together() {
        let (budget, runs) = (Cell::new(100), Cell::new(0));
        let found = ddmin((0..10).collect(), &budget, needs(&[2, 7], &budget, &runs));
        assert_eq!(found, [2, 7]);
        // 1-minimal: dropping either one no longer fails.
        let mut check = needs(&[2, 7], &budget, &runs);
        assert!(check(&found));
        assert!(!check(&[2]));
        assert!(!check(&[7]));
    }

    #[test]
    fn ddmin_leaves_short_inputs_alone() {
        let (budget, runs) = (Cell::new(100), Cell::new(0));
        assert_eq!(ddmin(vec![4], &budget, needs(&[4], &budget, &runs)), [4]);
        assert_eq!(
            ddmin(Vec::new(), &budget, needs(&[], &budget, &runs)),
            [] as [u32; 0]
        );
        assert_eq!(runs.get(), 0);
    }

    #[test]
    fn ddmin_returns_everything_when_no_part_fails_alone() {
        // Nothing ever fails (say, no subset applies): the input comes back.
        let (budget, runs) = (Cell::new(100), Cell::new(0));
        let found = ddmin((0..4).collect(), &budget, |_: &[u32]| {
            runs.set(runs.get() + 1);
            false
        });
        assert_eq!(found, [0, 1, 2, 3]);
        assert!(runs.get() > 0);
    }

    #[test]
    fn ddmin_stops_when_the_budget_is_spent() {
        let (budget, runs) = (Cell::new(3), Cell::new(0));
        let cause = [2, 7, 11];
        let found = ddmin((0..16).collect(), &budget, needs(&cause, &budget, &runs));
        assert_eq!(runs.get(), 3, "one run per unit of budget");
        assert!(budget.get() <= 0);
        // Not minimal yet, but what is returned still fails.
        assert!(found.len() > cause.len());
        assert!(cause.iter().all(|item| found.contains(item)));

        let (budget, runs) = (Cell::new(0), Cell::new(0));
        let untouched = ddmin((0..4).collect(), &budget, needs(&[1], &budget, &runs));
        assert_eq!(untouched, [0, 1, 2, 3]);
        assert_eq!(runs.get(), 0);
    }
}
