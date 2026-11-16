//! Regenerate or check the Mattar–Daw results files.

#![forbid(unsafe_code)]

use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use strata_backtest::report::render_results;

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(err) => {
            eprintln!("strata-backtest: {err}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> Result<(), String> {
    let args = parse(env::args().skip(1))?;
    let prereg = fs::read(&args.prereg).map_err(|err| format!("read prereg: {err}"))?;
    let rendered = render_results(&prereg)?;
    if let Some(dir) = args.out {
        fs::create_dir_all(&dir).map_err(|err| format!("create out: {err}"))?;
        write(&dir.join("mattar-evb-v1.json"), &rendered.json)?;
        write(&dir.join("mattar-evb-v1.md"), &rendered.markdown)?;
        return Ok(());
    }
    let json_path = args.check.expect("parse requires --out or --check");
    let on_disk = fs::read(&json_path).map_err(|err| format!("read results json: {err}"))?;
    mismatch("json", &on_disk, rendered.json.as_bytes())?;
    let md_path = json_path
        .parent()
        .unwrap_or_else(|| Path::new("."))
        .join("mattar-evb-v1.md");
    let md = fs::read(&md_path).map_err(|err| format!("read results markdown: {err}"))?;
    mismatch("markdown", &md, rendered.markdown.as_bytes())?;
    Ok(())
}

fn write(path: &Path, bytes: &str) -> Result<(), String> {
    fs::write(path, bytes).map_err(|err| format!("write {}: {err}", path.display()))
}

fn mismatch(kind: &str, found: &[u8], expected: &[u8]) -> Result<(), String> {
    if found == expected {
        return Ok(());
    }
    let at = found
        .iter()
        .zip(expected.iter())
        .position(|(left, right)| left != right)
        .unwrap_or(found.len().min(expected.len()));
    Err(format!(
        "{kind} mismatch at byte {at} (disk {}, regenerated {})",
        found.len(),
        expected.len()
    ))
}

#[derive(Debug)]
struct Args {
    prereg: PathBuf,
    out: Option<PathBuf>,
    check: Option<PathBuf>,
}

fn parse(input: impl IntoIterator<Item = String>) -> Result<Args, String> {
    let mut prereg = None;
    let mut out = None;
    let mut check = None;
    let mut args = input.into_iter();
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--prereg" => prereg = Some(value(&mut args, "--prereg")?),
            "--out" => out = Some(value(&mut args, "--out")?),
            "--check" => check = Some(value(&mut args, "--check")?),
            other => return Err(format!("unknown argument {other}")),
        }
    }
    let prereg = prereg.ok_or("--prereg is required")?;
    match (out, check) {
        (Some(_), Some(_)) => Err("--out and --check are separate commands".into()),
        (None, None) => Err("--out or --check is required".into()),
        (out, check) => Ok(Args { prereg, out, check }),
    }
}

fn value(args: &mut impl Iterator<Item = String>, flag: &str) -> Result<PathBuf, String> {
    args.next()
        .map(PathBuf::from)
        .ok_or_else(|| format!("{flag} needs a path"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_rejects_a_mixed_command() {
        let err = parse([
            "--prereg".into(),
            "p".into(),
            "--out".into(),
            "o".into(),
            "--check".into(),
            "c".into(),
        ])
        .unwrap_err();
        assert!(err.contains("separate"));
    }
}
