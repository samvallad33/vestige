//! v3 importer. Holds the store lock across the staged import and the rename
//! onto `log/`. The v3 file is only read.

use std::env;
use std::fs::{self, File};
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use vestige_upgrade::{UpgradeStatus, upgrade_if_needed};

fn main() -> ExitCode {
    let mut data_dir: Option<PathBuf> = None;
    let mut args = env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--help" | "-h" => {
                println!(
                    "vestige-upgrade --data-dir <DIR>\n\n\
                     Import <DIR>/vestige.db into a Strata log. The v3 file is not modified."
                );
                return ExitCode::SUCCESS;
            }
            "--version" | "-V" => {
                println!("vestige-upgrade {}", env!("CARGO_PKG_VERSION"));
                return ExitCode::SUCCESS;
            }
            "--data-dir" => {
                let Some(value) = args.next() else {
                    eprintln!("vestige-upgrade: --data-dir needs a path");
                    return ExitCode::from(2);
                };
                data_dir = Some(PathBuf::from(value));
            }
            other => {
                eprintln!("vestige-upgrade: unknown argument {other}");
                return ExitCode::from(2);
            }
        }
    }

    let data_dir = data_dir.unwrap_or_else(default_data_dir);
    if let Err(err) = fs::create_dir_all(&data_dir) {
        eprintln!(
            "vestige-upgrade: failed to create {}: {err}",
            data_dir.display()
        );
        return ExitCode::from(1);
    }
    // Same exclusive lock `vestige-mcp` holds while serving. Released on exit,
    // including SIGKILL. The staging rename runs under this lock.
    let _store_lock = match hold_store_lock(&data_dir) {
        Ok(file) => file,
        Err(code) => return code,
    };

    let db_path = data_dir.join("vestige.db");
    match upgrade_if_needed(&db_path) {
        Ok(UpgradeStatus::NoV3 | UpgradeStatus::StrataReady { .. }) => ExitCode::SUCCESS,
        Err(err) => {
            eprintln!("{err}");
            let _ = std::io::Write::flush(&mut std::io::stderr());
            ExitCode::from(1)
        }
    }
}

fn default_data_dir() -> PathBuf {
    if let Some(value) = env::var_os("VESTIGE_DATA_DIR")
        && !value.is_empty()
    {
        return PathBuf::from(value);
    }
    directories::ProjectDirs::from("com", "vestige", "core")
        .map(|dirs| dirs.data_dir().to_path_buf())
        .unwrap_or_else(|| PathBuf::from("."))
}

fn hold_store_lock(data_dir: &Path) -> Result<File, ExitCode> {
    let path = data_dir.join(".serve.lock");
    let file = match File::options()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .open(&path)
    {
        Ok(file) => file,
        Err(err) => {
            eprintln!(
                "vestige-upgrade: failed to create {}: {err}",
                path.display()
            );
            return Err(ExitCode::from(1));
        }
    };
    if let Err(err) = file.lock() {
        eprintln!(
            "vestige-upgrade: failed to lock {} (another vestige process holds it): {err}",
            path.display()
        );
        return Err(ExitCode::from(1));
    }
    Ok(file)
}
