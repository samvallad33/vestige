//! `vestige-verify-proof` and `vestige verify-proof`: one argument parser for
//! both. Exit codes: 0 the bundle verified and the policy holds, 1 it did not
//! (a missing bundle included), 2 the arguments or a file could not be used.

use std::path::{Path, PathBuf};

use crate::config::WitnessConfig;
use crate::didkey::TrustedKey;
use crate::verify::{Policy, Report, render_markdown, render_text, verify_bundle};
use crate::{PROOF_DIR, bundle_file_name, git};

pub const USAGE: &str = "\
Verify a proof-carrying PR bundle offline. Never opens a Vestige store.

usage: vestige-verify-proof [options]

  --repo DIR          git checkout to read (default .)
  --commit REV        prove this commit's tree (default HEAD)
  --tree HEX          the tree to check, instead of reading one from git
  --bundle FILE       the bundle; default .vestige/proof/<tree>.intoto.jsonl,
                      then the git note under refs/notes/vestige-proof
  --notes             read the bundle from the git note only
  --trust DID         a trusted did:key (repeatable)
  --trust-file FILE   trusted did:keys, one per line, # comments
  --require CMD       a command that must be green (repeatable)
  --require-file FILE every command in a witness.toml must be green
  --base REV          read .vestige/trusted-keys and .vestige/witness.toml from
                      REV, the branch a pull request merges into, so the pull
                      request cannot change what it is checked against
  --strict            every allowlisted command in the bundle must be green
  --format FMT        text (default), json or markdown
  --summary FILE      also append the Markdown report to FILE
";

#[derive(Default)]
struct Args {
    repo: Option<PathBuf>,
    commit: Option<String>,
    tree: Option<String>,
    bundle: Option<PathBuf>,
    notes: bool,
    trust: Vec<String>,
    trust_files: Vec<PathBuf>,
    require: Vec<String>,
    require_files: Vec<PathBuf>,
    base: Option<String>,
    strict: bool,
    format: String,
    summary: Option<PathBuf>,
}

fn parse(argv: &[String]) -> Result<Args, String> {
    let mut a = Args {
        format: "text".into(),
        ..Args::default()
    };
    let mut it = argv.iter();
    while let Some(flag) = it.next() {
        let mut value = |name: &str| {
            it.next()
                .cloned()
                .ok_or_else(|| format!("{name} needs a value"))
        };
        match flag.as_str() {
            "--repo" => a.repo = Some(value("--repo")?.into()),
            "--commit" => a.commit = Some(value("--commit")?),
            "--tree" => a.tree = Some(value("--tree")?),
            "--bundle" => a.bundle = Some(value("--bundle")?.into()),
            "--notes" => a.notes = true,
            "--trust" => a.trust.push(value("--trust")?),
            "--trust-file" => a.trust_files.push(value("--trust-file")?.into()),
            "--require" => a.require.push(value("--require")?),
            "--require-file" => a.require_files.push(value("--require-file")?.into()),
            "--base" => a.base = Some(value("--base")?),
            "--strict" => a.strict = true,
            "--format" => a.format = value("--format")?,
            "--summary" => a.summary = Some(value("--summary")?.into()),
            "-h" | "--help" => return Err(String::new()),
            other => return Err(format!("unknown argument {other:?}")),
        }
    }
    if !matches!(a.format.as_str(), "text" | "json" | "markdown") {
        return Err(format!(
            "--format is text, json or markdown, not {:?}",
            a.format
        ));
    }
    if a.tree.is_some() && a.commit.is_some() {
        return Err("pass --tree or --commit, not both".into());
    }
    Ok(a)
}

fn read(path: &Path, what: &str) -> Result<String, String> {
    std::fs::read_to_string(path)
        .map_err(|e| format!("could not read {what} {}: {e}", path.display()))
}

/// Run the verifier with `argv` (the arguments after the program name),
/// printing to stdout and stderr, and return the exit code.
pub fn run(argv: &[String]) -> i32 {
    let a = match parse(argv) {
        Ok(a) => a,
        Err(e) if e.is_empty() => {
            print!("{USAGE}");
            return 0;
        }
        Err(e) => {
            eprintln!("vestige-verify-proof: {e}\n\n{USAGE}");
            return 2;
        }
    };
    match run_args(&a) {
        Ok((report, text)) => {
            print!("{text}");
            if let Some(path) = &a.summary {
                let md = render_markdown(&report);
                let appended = std::fs::OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(path)
                    .and_then(|mut f| std::io::Write::write_all(&mut f, md.as_bytes()));
                if let Err(e) = appended {
                    eprintln!("vestige-verify-proof: could not write the summary: {e}");
                }
            }
            if report.ok { 0 } else { 1 }
        }
        Err(e) => {
            eprintln!("vestige-verify-proof: {e}");
            2
        }
    }
}

fn run_args(a: &Args) -> Result<(Report, String), String> {
    let mut trusted = Vec::new();
    for did in &a.trust {
        trusted.push(TrustedKey::parse(did).map_err(|e| e.0)?);
    }
    for file in &a.trust_files {
        trusted.extend(TrustedKey::parse_list(&read(file, "trust file")?).map_err(|e| e.0)?);
    }
    let repo = a.repo.clone().unwrap_or_else(|| PathBuf::from("."));
    let mut base_require = Vec::new();
    if let Some(rev) = &a.base {
        let base = git::commit_id(&repo, rev).map_err(|e| format!("--base: {}", e.0))?;
        if let Some(text) = git::show_file(&repo, &base, crate::TRUSTED_KEYS) {
            trusted.extend(
                TrustedKey::parse_list(&text)
                    .map_err(|e| format!("{} on {rev}: {}", crate::TRUSTED_KEYS, e.0))?,
            );
        }
        if let Some(text) = git::show_file(&repo, &base, crate::REPO_CONFIG) {
            base_require = WitnessConfig::parse(&text).commands;
        }
    }
    if trusted.is_empty() {
        return Err(format!(
            "no trusted key: pass --trust did:key:..., --trust-file, or --base with {} on that branch",
            crate::TRUSTED_KEYS
        ));
    }
    let mut policy = Policy {
        require: a.require.clone(),
        strict: a.strict,
    };
    for c in base_require {
        if !policy.require.contains(&c) {
            policy.require.push(c);
        }
    }
    for file in &a.require_files {
        for c in WitnessConfig::parse(&read(file, "require file")?).commands {
            if !policy.require.contains(&c) {
                policy.require.push(c);
            }
        }
    }

    let mut commit = None;
    let tree = match &a.tree {
        Some(t) if crate::is_object_id(t) => t.clone(),
        Some(t) => return Err(format!("--tree {t:?} is not a git object id")),
        None if a.bundle.is_some() && a.commit.is_none() && !repo.join(".git").exists() => {
            String::new()
        }
        None => {
            let c =
                git::commit_id(&repo, a.commit.as_deref().unwrap_or("HEAD")).map_err(|e| e.0)?;
            let t = git::commit_tree(&repo, &c).map_err(|e| e.0)?;
            commit = Some(c);
            t
        }
    };

    let bytes = if let Some(path) = &a.bundle {
        Some(std::fs::read(path).map_err(|e| format!("could not read {}: {e}", path.display()))?)
    } else {
        let file = repo.join(PROOF_DIR).join(bundle_file_name(&tree));
        let from_file = (!a.notes && file.is_file())
            .then(|| std::fs::read(&file).ok())
            .flatten();
        from_file.or_else(|| {
            commit
                .as_deref()
                .and_then(|c| git::read_note(&repo, c))
                .map(|n| format!("{n}\n").into_bytes())
        })
    };
    let expect = (!tree.is_empty()).then_some(tree.as_str());
    let report = match bytes {
        Some(b) => verify_bundle(&b, &trusted, expect, &policy),
        None => Report {
            tree: Some(tree.clone()),
            commit,
            failures: vec![format!(
                "no proof bundle for tree {tree}: no {PROOF_DIR}/{} and no note under {}",
                bundle_file_name(&tree),
                crate::NOTES_REF
            )],
            ..Report::default()
        },
    };
    let text = match a.format.as_str() {
        "json" => format!(
            "{}\n",
            serde_json::to_string_pretty(&report).map_err(|e| e.to_string())?
        ),
        "markdown" => render_markdown(&report),
        _ => render_text(&report),
    };
    Ok((report, text))
}
