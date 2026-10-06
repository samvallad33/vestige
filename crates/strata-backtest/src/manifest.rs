//! Canonical harness manifest: preregistration hash and the signed head per cut.

use strata_store::{StoreError, StrataStore};

use crate::canon::{Json, hex_bytes};
use crate::cuts::{eligible_bounds, thin_cuts};
use crate::events::DreamWindow;
use crate::mechanism::admitted_events;
use crate::protocol::TABLE_ID;

/// One cut's signed head. Frame hashes are comparable only inside one seeded log.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CutHead {
    /// Eligible event seq the prefix was folded through.
    pub bound_seq: u64,
    /// Greatest folded frame seq (`as_of` head).
    pub prefix_head_seq: u64,
    /// `frame_hash` of [`Self::prefix_head_seq`].
    pub prefix_head_frame_hash: [u8; 32],
    /// `state_digest` of the prefix fold.
    pub state_digest: [u8; 32],
}

/// `blake3` of the preregistration bytes.
pub fn prereg_blake3(bytes: &[u8]) -> [u8; 32] {
    *blake3::hash(bytes).as_bytes()
}

/// Render the harness manifest. Keys are sorted. Trailing newline.
pub fn render_manifest(prereg: &[u8], cuts: &[CutHead], windows: &[DreamWindow]) -> String {
    let cut_json = cuts
        .iter()
        .map(|cut| {
            Json::obj(vec![
                ("bound_seq", Json::U64(cut.bound_seq)),
                (
                    "prefix_head_frame_hash",
                    Json::Str(hex_bytes(&cut.prefix_head_frame_hash)),
                ),
                ("prefix_head_seq", Json::U64(cut.prefix_head_seq)),
                ("state_digest", Json::Str(hex_bytes(&cut.state_digest))),
            ])
        })
        .collect();
    let window_json = windows
        .iter()
        .map(|window| {
            Json::obj(vec![
                ("end_seq", Json::U64(window.end_seq)),
                ("start_seq", Json::U64(window.start_seq)),
            ])
        })
        .collect();
    Json::obj(vec![
        ("cuts", Json::Arr(cut_json)),
        ("dream_windows", Json::Arr(window_json)),
        (
            "prereg_blake3",
            Json::Str(hex_bytes(&prereg_blake3(prereg))),
        ),
        ("table_id", Json::Str(TABLE_ID.into())),
    ])
    .canonical()
}

/// Copy-only entry: fold each thinned cut and pin its signed head.
///
/// `store` must be the copy. This function only calls `as_of`.
pub fn harness_manifest(
    store: &StrataStore,
    prereg: &[u8],
    windows: &[DreamWindow],
) -> Result<String, StoreError> {
    let events = admitted_events(store)?;
    let bounds = thin_cuts(&eligible_bounds(&events, windows));
    let mut cuts = Vec::with_capacity(bounds.len());
    for bound in bounds {
        let fold = store.as_of(bound)?;
        cuts.push(CutHead {
            bound_seq: bound,
            prefix_head_seq: fold.prefix_head_seq(),
            prefix_head_frame_hash: fold.prefix_head_frame_hash(),
            state_digest: fold.state_digest(),
        });
    }
    Ok(render_manifest(prereg, &cuts, windows))
}
