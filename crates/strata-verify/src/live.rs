//! Read-only check of a live strata-store directory: `log/*.seg` at the
//! root. `store.meta` is present only after a checkpoint seal. An unsealed
//! head is still a live store. The root is not opened as a log, so this
//! never mints `strata.key` or a segment beside `store.meta`.

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
    let meta = dir.join(META_NAME);
    if meta.is_file() {
        match fs::read(&meta) {
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
