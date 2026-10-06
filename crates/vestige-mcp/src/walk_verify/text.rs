//! Output colors, refusals, paths, and the text helpers that keep Python's
//! meaning of "line", "strip", "[:n]" and "%g".

use std::fmt;
use std::io::IsTerminal;
use std::path::{Component, Path, PathBuf};

use anyhow::Context;
use chrono::Utc;
use serde_json::Value;

use super::json::canonical_json;

#[derive(Clone, Copy)]
pub(super) struct Palette {
    pub b: &'static str,
    pub d: &'static str,
    pub c: &'static str,
    pub g: &'static str,
    pub r: &'static str,
    pub y: &'static str,
    pub o: &'static str,
}

const COLOR: Palette = Palette {
    b: "\x1b[1m",
    d: "\x1b[2m",
    c: "\x1b[1;36m",
    g: "\x1b[1;32m",
    r: "\x1b[1;31m",
    y: "\x1b[1;33m",
    o: "\x1b[0m",
};

const PLAIN: Palette = Palette {
    b: "",
    d: "",
    c: "",
    g: "",
    r: "",
    y: "",
    o: "",
};

impl Palette {
    /// Color on a terminal or under `FORCE_COLOR`; never under `NO_COLOR`.
    pub fn detect() -> Self {
        let set = |name: &str| std::env::var_os(name).is_some_and(|value| !value.is_empty());
        if set("FORCE_COLOR") {
            COLOR
        } else if set("NO_COLOR") || !std::io::stdout().is_terminal() {
            PLAIN
        } else {
            COLOR
        }
    }

    pub fn verdict(&self, verdict: &str) -> &'static str {
        match verdict {
            "good" => self.g,
            "bad" => self.r,
            _ => self.y,
        }
    }
}

/// A refusal with its own exit code. The message is the whole output.
#[derive(Debug)]
pub(super) struct Stop {
    pub code: i32,
    pub message: String,
}

/// An error that ends the run with `code` after printing `message`.
pub(super) fn stop(code: i32, message: impl Into<String>) -> anyhow::Error {
    anyhow::Error::new(Stop {
        code,
        message: message.into(),
    })
}

impl fmt::Display for Stop {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for Stop {}

/// Now, to the second, in the form Python's
/// `datetime.now(timezone.utc).isoformat(timespec="seconds")` writes.
pub(super) fn now_stamp() -> String {
    Utc::now().format("%Y-%m-%dT%H:%M:%S+00:00").to_string()
}

/// The first `n` characters (not bytes) of `s`.
pub(super) fn head(s: &str, n: usize) -> &str {
    match s.char_indices().nth(n) {
        Some((index, _)) => &s[..index],
        None => s,
    }
}

fn is_line_break(c: char) -> bool {
    matches!(
        c,
        '\n' | '\r'
            | '\x0b'
            | '\x0c'
            | '\x1c'
            | '\x1d'
            | '\x1e'
            | '\u{85}'
            | '\u{2028}'
            | '\u{2029}'
    )
}

/// `str.splitlines()`: every line boundary Python knows, `\r\n` as one, and
/// no empty last line.
pub(super) fn split_lines(s: &str) -> Vec<&str> {
    let mut lines = Vec::new();
    let mut start = 0;
    let mut chars = s.char_indices().peekable();
    while let Some((index, c)) = chars.next() {
        if !is_line_break(c) {
            continue;
        }
        lines.push(&s[start..index]);
        start = index + c.len_utf8();
        if c == '\r'
            && let Some(&(next, '\n')) = chars.peek()
        {
            chars.next();
            start = next + 1;
        }
    }
    if start < s.len() {
        lines.push(&s[start..]);
    }
    lines
}

/// `str.strip()`.
pub(super) fn strip(s: &str) -> &str {
    s.trim_matches(|c: char| c.is_whitespace() || ('\x1c'..='\x1f').contains(&c))
}

/// The last line a test printed, cut to 160 characters.
pub(super) fn last_line(output: &[u8]) -> String {
    let text = String::from_utf8_lossy(output);
    split_lines(strip(&text))
        .last()
        .map(|line| head(line, 160).to_string())
        .unwrap_or_default()
}

/// How Python prints a JSON value with `%s`: `None` for null or missing.
pub(super) fn py_display(value: Option<&Value>) -> String {
    match value {
        None | Some(Value::Null) => "None".to_string(),
        Some(Value::String(text)) => text.clone(),
        Some(Value::Bool(true)) => "True".to_string(),
        Some(Value::Bool(false)) => "False".to_string(),
        Some(other) => canonical_json(other),
    }
}

pub(super) fn opt_display(value: Option<&str>) -> &str {
    value.unwrap_or("None")
}

/// C's `%.{precision}g`, which is Python's: `precision` significant digits,
/// plain decimals for an exponent from -4 up to the precision and
/// scientific notation with a two-digit exponent otherwise, trailing zeros
/// dropped.
pub(super) fn format_g(value: f64, precision: usize) -> String {
    if value.is_nan() {
        return "nan".to_string();
    }
    if value.is_infinite() {
        return if value > 0.0 { "inf" } else { "-inf" }.to_string();
    }
    if value == 0.0 {
        return if value.is_sign_negative() { "-0" } else { "0" }.to_string();
    }
    let precision = precision.max(1);
    // The exponent of the value once it is rounded to `precision` digits.
    let scientific = format!("{value:.digits$e}", digits = precision - 1);
    let (mantissa, exponent) = scientific.split_once('e').unwrap_or((&scientific, "0"));
    let exponent: i64 = exponent.parse().unwrap_or(0);
    let trimmed = |digits: &str| -> String {
        if digits.contains('.') {
            digits
                .trim_end_matches('0')
                .trim_end_matches('.')
                .to_string()
        } else {
            digits.to_string()
        }
    };
    if exponent < -4 || exponent >= precision as i64 {
        format!(
            "{}e{}{:02}",
            trimmed(mantissa),
            if exponent < 0 { '-' } else { '+' },
            exponent.unsigned_abs()
        )
    } else {
        let decimals = usize::try_from(precision as i64 - 1 - exponent).unwrap_or(0);
        trimmed(&format!("{value:.decimals$}"))
    }
}

/// `~` and `~/rest` under the home directory; anything else as given.
pub(super) fn expand_user(path: &Path) -> PathBuf {
    if let Ok(rest) = path.strip_prefix("~")
        && let Some(dirs) = directories::BaseDirs::new()
    {
        return dirs.home_dir().join(rest);
    }
    path.to_path_buf()
}

/// `os.path.abspath`: absolute and normalized, symlinks left alone.
pub(super) fn abspath(path: &Path) -> anyhow::Result<PathBuf> {
    let joined = if path.is_absolute() {
        path.to_path_buf()
    } else {
        std::env::current_dir()
            .context("cannot read the current directory")?
            .join(path)
    };
    let mut out = PathBuf::new();
    for component in joined.components() {
        match component {
            Component::CurDir => {}
            Component::ParentDir => {
                out.pop();
            }
            other => out.push(other.as_os_str()),
        }
    }
    Ok(out)
}

pub(super) fn utf8(path: &Path) -> anyhow::Result<&str> {
    path.to_str()
        .with_context(|| format!("path is not UTF-8: {}", path.display()))
}

pub(super) fn file_name(path: &Path) -> String {
    path.file_name()
        .map(|name| name.to_string_lossy().into_owned())
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn lines_and_cuts_follow_python() {
        assert_eq!(
            split_lines("a\nb\r\nc\rd\u{2028}e\n"),
            ["a", "b", "c", "d", "e"]
        );
        assert_eq!(split_lines("a\n\nb"), ["a", "", "b"]);
        assert!(split_lines("").is_empty());
        assert_eq!(head("h\u{e9}llo", 2), "h\u{e9}");
        assert_eq!(head("hi", 5), "hi");
        assert_eq!(head("", 3), "");
        assert_eq!(strip(" \t x y \x1c\n"), "x y");
        assert_eq!(last_line(b"one\ntwo\n"), "two");
        assert_eq!(last_line(b"one\nwarn\nlast  \n\n"), "last");
        assert_eq!(last_line(b"progress 1\rprogress 2"), "progress 2");
        assert_eq!(last_line(b""), "");
        assert_eq!(last_line(b" \n\n"), "");
        assert_eq!(last_line("x".repeat(200).as_bytes()).len(), 160);
        // Bytes that are not UTF-8 do not stop the run.
        assert_eq!(last_line(b"ok \xff\xfe"), "ok \u{fffd}\u{fffd}");
    }

    #[test]
    fn values_print_the_way_python_prints_them() {
        assert_eq!(py_display(None), "None");
        assert_eq!(py_display(Some(&Value::Null)), "None");
        assert_eq!(py_display(Some(&json!("text"))), "text");
        assert_eq!(py_display(Some(&json!(true))), "True");
        assert_eq!(py_display(Some(&json!(17))), "17");
        assert_eq!(opt_display(None), "None");
        assert_eq!(opt_display(Some("mem-01")), "mem-01");
    }

    #[test]
    fn percent_g_follows_c() {
        // "%.2g", "%.3g" and "%g" of each value, from Python.
        for (value, two, three, six) in [
            (0.0, "0", "0", "0"),
            (1.0, "1", "1", "1"),
            (0.001, "0.001", "0.001", "0.001"),
            (0.0009765625, "0.00098", "0.000977", "0.000976562"),
            (1.2345e-05, "1.2e-05", "1.23e-05", "1.2345e-05"),
            (1.5e-09, "1.5e-09", "1.5e-09", "1.5e-09"),
            (123456789.0, "1.2e+08", "1.23e+08", "1.23457e+08"),
            (0.5, "0.5", "0.5", "0.5"),
            (0.01, "0.01", "0.01", "0.01"),
            (0.05, "0.05", "0.05", "0.05"),
            (2.5e-06, "2.5e-06", "2.5e-06", "2.5e-06"),
            (9.999e-05, "0.0001", "0.0001", "9.999e-05"),
            (0.00012345, "0.00012", "0.000123", "0.00012345"),
            (99.5, "1e+02", "99.5", "99.5"),
            (0.995, "0.99", "0.995", "0.995"),
            (0.0001, "0.0001", "0.0001", "0.0001"),
            (12.0, "12", "12", "12"),
            (4.17e-06, "4.2e-06", "4.17e-06", "4.17e-06"),
            (1e16, "1e+16", "1e+16", "1e+16"),
            (3.14159e-12, "3.1e-12", "3.14e-12", "3.14159e-12"),
            (0.000123456789, "0.00012", "0.000123", "0.000123457"),
            (100.0, "1e+02", "100", "100"),
            (1e-10, "1e-10", "1e-10", "1e-10"),
            (0.0003985065657050695, "0.0004", "0.000399", "0.000398507"),
            (8.884673390211095e-06, "8.9e-06", "8.88e-06", "8.88467e-06"),
        ] {
            assert_eq!(format_g(value, 2), two, "%.2g of {value:e}");
            assert_eq!(format_g(value, 3), three, "%.3g of {value:e}");
            assert_eq!(format_g(value, 6), six, "%g of {value:e}");
        }
        assert_eq!(format_g(-0.5, 2), "-0.5");
        assert_eq!(format_g(f64::NAN, 2), "nan");
        assert_eq!(format_g(f64::INFINITY, 2), "inf");
    }

    #[test]
    fn paths_are_made_absolute_without_touching_symlinks() {
        assert_eq!(
            abspath(Path::new("/a/./b/../c")).unwrap(),
            Path::new("/a/c")
        );
        let relative = abspath(Path::new("x/y.json")).unwrap();
        assert!(relative.is_absolute());
        assert!(relative.ends_with("x/y.json"));
        assert_eq!(
            abspath(Path::new("/with space/r e p o")).unwrap(),
            Path::new("/with space/r e p o")
        );
        assert_eq!(file_name(Path::new("/a/b c.json")), "b c.json");
        assert_eq!(file_name(Path::new("/")), "");
        assert_eq!(expand_user(Path::new("/abs/x")), Path::new("/abs/x"));
        assert_eq!(expand_user(Path::new("rel/~x")), Path::new("rel/~x"));
    }

    #[test]
    fn the_time_stamp_is_utc_to_the_second() {
        let stamp = now_stamp();
        assert_eq!(stamp.len(), 25, "{stamp}");
        assert!(stamp.ends_with("+00:00"), "{stamp}");
        assert!(chrono::DateTime::parse_from_rfc3339(&stamp).is_ok());
    }
}
