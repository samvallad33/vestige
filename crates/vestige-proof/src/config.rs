//! The witness allowlist.
//!
//! Two files use the same two tables: the owner's `<data dir>/vestige.toml`
//! and a repository's `.vestige/witness.toml`.
//!
//! ```toml
//! [witness]
//! commands = ["cargo test --workspace", "pnpm test"]
//!
//! [witness.junit]
//! "cargo nextest run --profile ci" = "target/nextest/ci/junit.xml"
//! ```
//!
//! A command is witnessed only when the whole command the agent ran equals one
//! entry exactly, after trimming surrounding whitespace. `cargo test | tail`
//! is a different command: its exit code is `tail`'s. Like the rest of
//! `vestige.toml` the reader is a small TOML subset: tables, string arrays,
//! basic and literal strings. Unknown tables and keys are ignored.

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct WitnessConfig {
    /// Exact commands, in file order, without duplicates.
    pub commands: Vec<String>,
    /// A junit report path, relative to the repository root, per command.
    pub junit: Vec<(String, String)>,
}

impl WitnessConfig {
    pub fn is_allowed(&self, command: &str) -> bool {
        let c = command.trim();
        !c.is_empty() && self.commands.iter().any(|a| a == c)
    }

    pub fn junit_for(&self, command: &str) -> Option<&str> {
        let c = command.trim();
        self.junit
            .iter()
            .find(|(k, _)| k == c)
            .map(|(_, v)| v.as_str())
    }

    /// Both lists of `other` added after this one's, without duplicates.
    pub fn merge(mut self, other: WitnessConfig) -> WitnessConfig {
        for c in other.commands {
            if !self.commands.contains(&c) {
                self.commands.push(c);
            }
        }
        for (k, v) in other.junit {
            if !self.junit.iter().any(|(x, _)| *x == k) {
                self.junit.push((k, v));
            }
        }
        self
    }

    pub fn parse(text: &str) -> WitnessConfig {
        let mut out = WitnessConfig::default();
        let mut table = String::new();
        let mut lines = text.lines();
        while let Some(raw) = lines.next() {
            let line = raw.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            if line.starts_with('[') && !line.starts_with("[[") {
                if let Some(end) = line.find(']') {
                    table = line[1..end].trim().to_string();
                }
                continue;
            }
            match table.as_str() {
                "witness" => {
                    let Some((key, rest)) = line.split_once('=') else {
                        continue;
                    };
                    if key.trim() != "commands" {
                        continue;
                    }
                    let mut value = rest.trim().to_string();
                    // An array may run over several lines.
                    while value.starts_with('[') && !array_closed(&value) {
                        let Some(next) = lines.next() else { break };
                        value.push('\n');
                        value.push_str(next.trim());
                    }
                    for c in strings_in_array(&value) {
                        let c = c.trim().to_string();
                        if !c.is_empty() && !out.commands.contains(&c) {
                            out.commands.push(c);
                        }
                    }
                }
                "witness.junit" => {
                    let Some((key, after)) = take_string(line) else {
                        continue;
                    };
                    let after = after.trim_start();
                    let Some(after) = after.strip_prefix('=') else {
                        continue;
                    };
                    let Some((path, _)) = take_string(after.trim_start()) else {
                        continue;
                    };
                    let key = key.trim().to_string();
                    if !key.is_empty()
                        && !path.is_empty()
                        && !out.junit.iter().any(|(k, _)| *k == key)
                    {
                        out.junit.push((key, path));
                    }
                }
                _ => {}
            }
        }
        out
    }
}

/// Read one TOML basic (`"…"`) or literal (`'…'`) string from the start of
/// `s`, returning it and the text after it.
fn take_string(s: &str) -> Option<(String, &str)> {
    let mut chars = s.char_indices();
    let (_, quote) = chars.next()?;
    if quote != '"' && quote != '\'' {
        return None;
    }
    let mut out = String::new();
    let mut escaped = false;
    for (i, c) in chars {
        if quote == '"' && escaped {
            out.push(match c {
                'n' => '\n',
                't' => '\t',
                'r' => '\r',
                other => other,
            });
            escaped = false;
        } else if quote == '"' && c == '\\' {
            escaped = true;
        } else if c == quote {
            return Some((out, &s[i + c.len_utf8()..]));
        } else {
            out.push(c);
        }
    }
    None
}

fn strings_in_array(value: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut rest = value.trim().strip_prefix('[').unwrap_or("");
    loop {
        rest = rest.trim_start_matches(|c: char| c.is_whitespace() || c == ',');
        if rest.starts_with('#') {
            rest = rest.split_once('\n').map(|(_, r)| r).unwrap_or("");
            continue;
        }
        match take_string(rest) {
            Some((s, after)) => {
                out.push(s);
                rest = after;
            }
            None => return out,
        }
    }
}

fn array_closed(value: &str) -> bool {
    let mut rest = value.trim().strip_prefix('[').unwrap_or("");
    loop {
        rest = rest.trim_start_matches(|c: char| c.is_whitespace() || c == ',');
        if rest.starts_with(']') {
            return true;
        }
        if rest.starts_with('#') {
            match rest.split_once('\n') {
                Some((_, r)) => rest = r,
                None => return false,
            }
            continue;
        }
        match take_string(rest) {
            Some((_, after)) => rest = after,
            None => return false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn commands_and_junit_paths_are_read_exactly() {
        let cfg = WitnessConfig::parse(
            r#"
            [defaults]
            profile = "lean"

            [witness]
            commands = ["cargo test --workspace", 'pnpm test # not a comment', "cargo test --workspace"]

            [witness.junit]
            "cargo nextest run --profile ci" = "target/nextest/ci/junit.xml"
            "#,
        );
        assert_eq!(
            cfg.commands,
            vec!["cargo test --workspace", "pnpm test # not a comment"]
        );
        assert_eq!(
            cfg.junit_for("cargo nextest run --profile ci"),
            Some("target/nextest/ci/junit.xml")
        );
        assert!(cfg.is_allowed("  cargo test --workspace\n"));
        assert!(!cfg.is_allowed("cargo test --workspace | tail -5"));
        assert!(!cfg.is_allowed("cargo test"));
    }

    #[test]
    fn a_multi_line_array_with_comments_and_escapes_is_read() {
        let cfg = WitnessConfig::parse(
            "[witness]\ncommands = [\n  \"go test ./...\",  # unit\n  \"npm test -- --grep \\\"api\\\"\",\n]\n",
        );
        assert_eq!(
            cfg.commands,
            vec!["go test ./...", "npm test -- --grep \"api\""]
        );
    }

    #[test]
    fn the_repository_list_adds_to_the_owner_list() {
        let owner = WitnessConfig::parse("[witness]\ncommands = [\"a\", \"b\"]");
        let repo = WitnessConfig::parse("[witness]\ncommands = [\"b\", \"c\"]");
        assert_eq!(owner.merge(repo).commands, vec!["a", "b", "c"]);
        assert_eq!(WitnessConfig::parse(""), WitnessConfig::default());
    }
}
