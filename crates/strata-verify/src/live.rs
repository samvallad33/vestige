//! Read-only check of a live strata-store directory: `log/*.seg`, and
//! `store.meta` when a checkpoint has been sealed. An unsealed store has
//! no anchor file; the segment chain is still checked. The root is not
//! opened as a log, so this never mints `strata.key` or a segment.

use std::fs;
use std::path::Path;

use serde::Serialize;

use crate::readonly;

const META_NAME: &str = "store.meta";
const LOG_DIR: &str = "log";
const META_MAGIC: [u8; 8] = *b"STRSTME1";

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub(crate) struct LiveVerifyReport {
    pub ok: bool,
    pub frames_total: u64,
    pub segments: u32,
    pub failures: Vec<String>,
}

pub(crate) fn is_live_store(dir: &Path) -> bool {
    // A fresh store has segments and no anchor until the first seal.
    readonly::dir_has_segments(&dir.join(LOG_DIR))
}

pub(crate) fn verify_live_store(dir: &Path) -> LiveVerifyReport {
    let mut failures = Vec::new();
    let scan = match readonly::scan_log(&dir.join(LOG_DIR)) {
        Ok(scan) => Some(scan),
        Err(err) => {
            failures.push(err);
            None
        }
    };
    let meta_path = dir.join(META_NAME);
    if meta_path.is_file() {
        match fs::read(&meta_path) {
            Ok(bytes) => {
                if bytes.len() < META_MAGIC.len() || bytes[..META_MAGIC.len()] != META_MAGIC {
                    failures.push("store.meta magic is not STRSTME1".into());
                }
            }
            Err(err) => failures.push(format!("read store.meta: {err}")),
        }
    }
    let (frames_total, segments) = scan
        .map(|scan| (scan.frames.len() as u64, scan.segments))
        .unwrap_or((0, 0));
    LiveVerifyReport {
        ok: failures.is_empty(),
        frames_total,
        segments,
        failures,
    }
}
