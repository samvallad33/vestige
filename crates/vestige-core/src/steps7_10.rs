// Computer 2 lane: Steps 7-10 helpers (included from retrieval.rs right after rungs.rs).
// Ingest-time extraction only reads `prose_body`; ranking helpers never read `TurnIx.body`
// (empty under `--no-dump`). Closed-class grammar only; no domain word lists.

/// Sam's 7B rare-modifier idf threshold.
const STEP7_RARE_IDF: f32 = 2.2;

/// BM25 idf over turn-own stats, `((n - df + 0.5) / (df + 0.5) + 1).ln().max(0)`.
fn step7_idf(stats: &Stats, term: u32) -> f32 {
    let df = stats.df.get(&term).copied().unwrap_or(0) as f32;
    ((stats.n - df + 0.5) / (df + 0.5) + 1.0).ln().max(0.0)
}

/// Lowercased query words written capitalized somewhere other than sentence start.
fn step7_query_proper_surfaces(q_text: &str) -> HashSet<String> {
    let mut out = HashSet::new();
    let mut sentence_start = true;
    for raw_word in q_text.split_whitespace() {
        let word = raw_word.trim_matches(|c: char| !c.is_alphanumeric());
        let word = word.strip_suffix("'s").or_else(|| word.strip_suffix("’s")).unwrap_or(word);
        if !sentence_start && word.chars().next().is_some_and(|c| c.is_uppercase()) {
            out.insert(word.to_lowercase());
        }
        sentence_start = raw_word.ends_with(['.', '!', '?']);
    }
    out
}

/// Step 7B (Lex specialis) at HOOK B: when the query carries a rare modifier group, sessions
/// that contain it damp their turns that lack it, so the specific rule outranks the general.
fn apply_step7_lex_specialis(
    ix: &Index,
    q_text: &str,
    plan: &QueryPlan,
    pref_on: bool,
    it: &QueryIntent,
    t_own: &[f32],
    t_scene: &mut [f32],
) {
    if pref_on || it.is_event_sequence || plan.groups.len() < 3 {
        return;
    }
    // v6: the "special law" is a named referent: only proper-name query groups compete.
    let proper = step7_query_proper_surfaces(q_text);
    let mut top_mod: Option<(usize, f32)> = None;
    for (g_idx, g_surf) in plan.groups.iter().enumerate() {
        let gl = g_surf.to_lowercase();
        if gl.len() < 3 || stopwords().contains(gl.as_str()) {
            continue;
        }
        let surf = gl.strip_prefix('\u{1}').and_then(|r| r.strip_prefix("ent:")).unwrap_or(&gl);
        if !surf.split('_').all(|part| proper.contains(part)) {
            continue;
        }
        let max_g_idf = plan
            .terms
            .iter()
            .filter(|qt| qt.group == Some(g_idx))
            .map(|qt| step7_idf(&ix.t_own_stats, qt.id))
            .fold(0.0_f32, f32::max);
        if max_g_idf >= STEP7_RARE_IDF && top_mod.is_none_or(|(_, best)| max_g_idf > best) {
            top_mod = Some((g_idx, max_g_idf));
        }
    }
    let Some((mod_g, _)) = top_mod else {
        return;
    };
    let mod_term_ids: HashSet<u32> =
        plan.terms.iter().filter(|qt| qt.group == Some(mod_g)).map(|qt| qt.id).collect();
    let turn_has_mod: Vec<bool> = ix
        .turns
        .iter()
        .map(|t| {
            t.feats.iter().any(|f| f.counts_len && mod_term_ids.contains(&f.id))
                || t.content.iter().any(|id| mod_term_ids.contains(id))
                // v2: a turn spoken by the modifier (named speaker) carries it by identity.
                || raw_tokens(&t.speaker)
                    .iter()
                    .any(|w| ix.vocab.get(w).is_some_and(|id| mod_term_ids.contains(&id)))
        })
        .collect();
    // v8: precedence, not damping. Inside a session that states the special rule, a general
    // turn may not outrank the best special turn; general turns already below it are untouched.
    for sess_turns in &ix.session_turns {
        let best_special = sess_turns
            .iter()
            .filter(|&&ti| !ix.turns[ti].is_question && turn_has_mod[ti] && t_own[ti] > 0.0)
            .map(|&ti| t_scene[ti])
            .fold(0.0_f32, f32::max);
        if best_special <= 0.0 {
            continue;
        }
        for &ti in sess_turns {
            if !turn_has_mod[ti] && t_scene[ti] > best_special {
                t_scene[ti] = (0.82 * t_scene[ti]).max(best_special);
            }
        }
    }
}

#[cfg(test)]
mod steps7_10_tests {
    use super::*;

    #[test]
    fn step7_proper_surfaces_skip_sentence_start() {
        let p = step7_query_proper_surfaces("What did Melanie's friend Jon say? Where is Rome");
        assert!(p.contains("melanie") && p.contains("jon"));
        assert!(!p.contains("where") && p.contains("rome"));
        assert!(!p.contains("what"));
    }
}
