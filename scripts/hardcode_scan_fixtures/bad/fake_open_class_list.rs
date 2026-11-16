// Negative fixture for scripts/hardcode_scan_selftest.sh. Not compiled and
// outside crates/, so the real scan never sees it. The scan MUST reject it.
const FAKE_VERB_CUES: &[&str] = &[
    "recommended", "suggested", "mentioned", "provided", "listed", "described",
    "explained", "calculated", "advised", "picked",
];
