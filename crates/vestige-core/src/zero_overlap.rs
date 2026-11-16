//! Zero-overlap gate for query expansion.
//!
//! This module used to hold a hand-built gazetteer (`CONCEPT_TABLE`: countries,
//! cities, landmarks, foods, sports, pets) plus taxonomy and allergen bitmasks
//! that let a query with no shared words reach a turn through a fixed concept
//! table. A fixed table of world knowledge is a word list, so it was removed
//! together with the bridge that read it.
//!
//! What remains is the pre-retrieval gate (after Ounis' query-performance
//! prediction gate): expansion may run only when the query, as typed, has zero
//! lexical overlap with the indexed corpus. It reads document frequencies
//! only; it holds no words.

/// True when no original query term occurs in the corpus (every df is 0, or
/// the query produced no corpus terms at all). Any overlap closes the gate,
/// so a normal query is never expanded.
pub(crate) fn ounis_zero_overlap_gate(query_term_dfs: impl IntoIterator<Item = u32>) -> bool {
    query_term_dfs.into_iter().all(|df| df == 0)
}

/// True when an individual query token is rare enough in the corpus
/// (`df <= max_rare_df`) to seed the isolated bridge channel.
pub(crate) fn term_rarity_gate(df: u32, max_rare_df: u32) -> bool {
    df <= max_rare_df
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gate_opens_only_on_zero_overlap() {
        assert!(ounis_zero_overlap_gate([]));
        assert!(ounis_zero_overlap_gate([0, 0]));
        assert!(!ounis_zero_overlap_gate([0, 3]));
    }
}
