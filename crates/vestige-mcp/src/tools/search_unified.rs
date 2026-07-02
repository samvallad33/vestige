//! Unified Search Tool
//!
//! Merges recall, semantic_search, and hybrid_search into a single `search` tool.
//! Always uses hybrid search internally (keyword + semantic + RRF fusion).
//! Implements Testing Effect (Roediger & Karpicke 2006) by auto-strengthening memories on access.
//!
//! v1.5.0: Enhanced 7-stage cognitive pipeline:
//!   1. Reranker (over-fetch 3x, rerank down)
//!   2. Temporal boosting (recency + validity)
//!   3. Memory state accessibility filtering
//!   4. Context matching (topic overlap)
//!   5. Spreading activation associations
//!   6. Predictive memory recording
//!   7. Reconsolidation (mark labile)

use chrono::Utc;
use serde::Deserialize;
use serde_json::Value;
use std::sync::Arc;
use tokio::sync::Mutex;

use crate::cognitive::CognitiveEngine;
use vestige_core::{
    CompetitionCandidate, EncodingContext, MemoryLifecycle, MemorySnapshot, MemoryState,
    OutputConfig, Storage, TopicalContext,
};

/// Input schema for unified search tool
pub fn schema() -> Value {
    serde_json::json!({
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search query"
            },
            "limit": {
                "type": "integer",
                "description": "Maximum number of results (default: 10)",
                "default": 10,
                "minimum": 1,
                "maximum": 100
            },
            "min_retention": {
                "type": "number",
                "description": "Minimum retention strength (0.0-1.0, default: 0.0)",
                "default": 0.0,
                "minimum": 0.0,
                "maximum": 1.0
            },
            "min_similarity": {
                "type": "number",
                "description": "Minimum similarity threshold (0.0-1.0, default: 0.5)",
                "default": 0.5,
                "minimum": 0.0,
                "maximum": 1.0
            },
            "detail_level": {
                "type": "string",
                "description": "Level of detail in results. 'brief' = id/type/tags/score only (saves tokens). 'summary' = default 8-field response. 'full' = all fields including FSRS state and timestamps.",
                "enum": ["brief", "summary", "full"],
                "default": "summary"
            },
            "context_topics": {
                "type": "array",
                "items": { "type": "string" },
                "description": "Optional topics for context-dependent retrieval boosting"
            },
            "exclude_types": {
                "type": "array",
                "items": { "type": "string" },
                "description": "Node types to exclude from results (e.g., ['reflection']). Reflections are excluded by default to prevent polluting factual queries."
            },
            "include_types": {
                "type": "array",
                "items": { "type": "string" },
                "description": "If set, only return nodes of these types. Overrides exclude_types."
            },
            "token_budget": {
                "type": "integer",
                "description": "Max tokens for response. Server truncates content to fit budget. Use memory(action='get') for full content of specific IDs. With 1M context models, budgets up to 100K are practical.",
                "minimum": 100,
                "maximum": 100000
            },
            "retrieval_mode": {
                "type": "string",
                "description": "precise: top results only (fast, token-efficient, skips activation/competition). balanced: full 7-stage cognitive pipeline (default). exhaustive: maximum recall with 5x overfetch, deep graph traversal, no competition suppression.",
                "enum": ["precise", "balanced", "exhaustive"],
                "default": "balanced"
            },
            "concrete": {
                "type": "boolean",
                "description": "Force literal/concrete search. Skips semantic expansion, FSRS reweighting, spreading activation, and cognitive side effects. Auto-enabled for quoted strings, env vars, UUIDs, paths, and code identifiers.",
                "default": false
            },
            "tag_prefix": {
                "type": "string",
                "description": "Optional tag-prefix filter. When set, only results carrying at least one tag whose value starts with this prefix are returned (case-sensitive). Example: tag_prefix=\"meeting:\" matches memories tagged 'meeting:standup', 'meeting:1-on-1', etc. Applied as a post-filter; combine with a larger 'limit' if you expect heavy thinning."
            },
            "source_system": {
                "type": "string",
                "description": "Investigation filter (#57): only memories ingested from this external system, e.g. 'github' or 'redmine'. Post-filter — non-connector memories are excluded. Combine with a larger 'limit' if thinning is heavy."
            },
            "source_project": {
                "type": "string",
                "description": "Investigation filter: only memories from this source project/repo, exact match (GitHub 'owner/repo', Redmine project id)."
            },
            "source_id": {
                "type": "string",
                "description": "Investigation filter: a specific source record id (issue number / ticket id). Pair with source_system to disambiguate across systems."
            },
            "source_type": {
                "type": "string",
                "description": "Investigation filter: source record type, e.g. 'issue', 'comment'."
            },
            "source_author": {
                "type": "string",
                "description": "Investigation filter: the source author/reporter (not assignee)."
            },
            "source_updated_after": {
                "type": "string",
                "description": "Investigation filter: only records whose source was updated at/after this RFC3339 timestamp (inclusive)."
            },
            "source_updated_before": {
                "type": "string",
                "description": "Investigation filter: only records whose source was updated at/before this RFC3339 timestamp (inclusive)."
            },
            "source_status": {
                "type": "string",
                "enum": ["any", "valid", "tombstoned"],
                "description": "Investigation filter: 'any' (default), 'valid' (currently-valid records only), or 'tombstoned' (records no longer visible upstream, kept for audit).",
                "default": "any"
            }
        },
        "required": ["query"]
    })
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct SearchArgs {
    query: String,
    limit: Option<i32>,
    #[serde(alias = "min_retention")]
    min_retention: Option<f64>,
    #[serde(alias = "min_similarity")]
    min_similarity: Option<f32>,
    #[serde(alias = "detail_level")]
    detail_level: Option<String>,
    #[serde(alias = "context_topics")]
    context_topics: Option<Vec<String>>,
    #[serde(alias = "exclude_types")]
    exclude_types: Option<Vec<String>>,
    #[serde(alias = "include_types")]
    include_types: Option<Vec<String>>,
    #[serde(alias = "token_budget")]
    token_budget: Option<i32>,
    #[serde(alias = "retrieval_mode")]
    retrieval_mode: Option<String>,
    concrete: Option<bool>,
    #[serde(alias = "tag_prefix")]
    tag_prefix: Option<String>,
    // #57 Phase 4 — source-aware investigation filters (all post-filters).
    #[serde(alias = "source_system")]
    source_system: Option<String>,
    #[serde(alias = "source_project")]
    source_project: Option<String>,
    #[serde(alias = "source_id")]
    source_id: Option<String>,
    #[serde(alias = "source_type")]
    source_type: Option<String>,
    #[serde(alias = "source_author")]
    source_author: Option<String>,
    #[serde(alias = "source_updated_after")]
    source_updated_after: Option<String>,
    #[serde(alias = "source_updated_before")]
    source_updated_before: Option<String>,
    #[serde(alias = "source_status")]
    source_status: Option<String>,
}

/// Execute unified search with 7-stage cognitive pipeline.
///
/// Pipeline:
///   1. Hybrid search (keyword + semantic + RRF) with 3x over-fetch
///   2. Reranker (BM25-like rescoring, trim to limit)
///   3. Temporal boosting (recency + validity windows)
///   4. Memory state accessibility filtering (Active/Dormant/Silent/Unavailable)
///   5. Context matching (topic overlap boosting)
///   6. Spreading activation (find associated memories)
///   7. Side effects: predictive memory recording + reconsolidation labile marking
///
/// Also applies Testing Effect (Roediger & Karpicke 2006) by auto-strengthening on access.
pub async fn execute(
    storage: &Arc<Storage>,
    cognitive: &Arc<Mutex<CognitiveEngine>>,
    output_config: &OutputConfig,
    args: Option<Value>,
) -> Result<Value, String> {
    let args: SearchArgs = match args {
        Some(v) => serde_json::from_value(v).map_err(|e| format!("Invalid arguments: {}", e))?,
        None => return Err("Missing arguments".to_string()),
    };

    if args.query.trim().is_empty() {
        return Err("Query cannot be empty".to_string());
    }

    // Validate detail_level. Precedence: explicit MCP param > config file >
    // built-in default. The explicit arg is validated; the config fallback is
    // already validated at load time.
    let detail_level_owned = output_config.resolve_detail_level(args.detail_level.as_deref());
    let detail_level = match detail_level_owned.as_str() {
        "brief" => "brief",
        "full" => "full",
        "summary" => "summary",
        invalid => {
            return Err(format!(
                "Invalid detail_level '{}'. Must be 'brief', 'summary', or 'full'.",
                invalid
            ));
        }
    };

    // Clamp all parameters to valid ranges. The default limit honors the
    // config file (e.g. a `research` profile) when no explicit param is set.
    let limit = output_config.resolve_limit(args.limit, 10).clamp(1, 100);
    let min_retention = args.min_retention.unwrap_or(0.0).clamp(0.0, 1.0);
    let min_similarity = args.min_similarity.unwrap_or(0.5).clamp(0.0, 1.0);

    // Validate retrieval_mode
    let retrieval_mode = match args.retrieval_mode.as_deref() {
        Some("precise") => "precise",
        Some("exhaustive") => "exhaustive",
        Some("balanced") | None => "balanced",
        Some(invalid) => {
            return Err(format!(
                "Invalid retrieval_mode '{}'. Must be 'precise', 'balanced', or 'exhaustive'.",
                invalid
            ));
        }
    };

    // #57 Phase 4 — parse the source-aware investigation filter once (shared by
    // both the concrete and hybrid paths). Hard-errors on malformed input.
    let source_filter = SourceFilter::from_args(&args)?;

    let concrete = args
        .concrete
        .unwrap_or_else(|| is_literal_query(&args.query));
    if concrete {
        // When a tag_prefix OR a source filter is requested, fetch a larger
        // pool so the post-filter has enough headroom to still return ~limit
        // results after thinning. Cap at the same upper bound the underlying
        // SQL path uses elsewhere (100).
        let concrete_fetch_limit = if args.tag_prefix.is_some() || source_filter.is_active() {
            (limit * 3).min(100)
        } else {
            limit
        };
        let results = storage
            .concrete_search_filtered(
                &args.query,
                concrete_fetch_limit,
                args.include_types.as_deref(),
                args.exclude_types.as_deref(),
            )
            .map_err(|e| e.to_string())?;

        // Apply tag_prefix post-filter BEFORE strengthen-on-access so
        // results the caller did not actually receive do not get a
        // testing-effect boost.
        let filtered_results: Vec<&vestige_core::SearchResult> = results
            .iter()
            .filter(|r| match args.tag_prefix.as_deref() {
                Some(prefix) => tags_match_prefix(&r.node.tags, prefix),
                None => true,
            })
            .filter(|r| node_matches_source(&r.node, &source_filter))
            .take(limit as usize)
            .collect();

        let ids: Vec<&str> = filtered_results
            .iter()
            .map(|r| r.node.id.as_str())
            .collect();
        let _ = storage.strengthen_batch_on_access(&ids);

        let mut formatted: Vec<Value> = filtered_results
            .iter()
            .filter(|r| r.node.retention_strength >= min_retention)
            .map(|r| format_search_result(r, detail_level))
            .collect();
        apply_output_masks(&mut formatted, output_config);

        let mut budget_expandable: Vec<String> = Vec::new();
        let mut budget_tokens_used: Option<usize> = None;
        if let Some(budget) = args.token_budget {
            let budget = budget.clamp(100, 100000) as usize;
            let budget_chars = budget * 4;
            let mut used = 0;
            let mut budgeted = Vec::new();

            for result in &formatted {
                let size = serde_json::to_string(result).unwrap_or_default().len();
                if used + size > budget_chars {
                    if let Some(id) = result.get("id").and_then(|v| v.as_str()) {
                        budget_expandable.push(id.to_string());
                    }
                    continue;
                }
                used += size;
                budgeted.push(result.clone());
            }

            budget_tokens_used = Some(used / 4);
            formatted = budgeted;
        }

        let mut response = serde_json::json!({
            "query": args.query,
            "method": "concrete",
            "retrievalMode": retrieval_mode,
            "concrete": true,
            "detailLevel": detail_level,
            "profile": output_config.profile.as_str(),
            "total": formatted.len(),
            "results": formatted,
        });

        if formatted.is_empty() {
            response["hint"] = serde_json::json!(
                "No concrete matches found. Try concrete=false or a broader natural-language query."
            );
        }
        if !budget_expandable.is_empty() {
            response["expandable"] = serde_json::json!(budget_expandable);
        }
        if let Some(tokens) = budget_tokens_used {
            response["tokenBudgetUsed"] = serde_json::json!(tokens);
            response["tokenBudgetLimit"] = serde_json::json!(args.token_budget.unwrap());
        }

        return Ok(response);
    }

    // Favor semantic search — research shows 0.3/0.7 outperforms equal weights
    let keyword_weight = 0.3_f32;
    let semantic_weight = 0.7_f32;

    // ====================================================================
    // STAGE 0: Keyword-first search (dedicated keyword-only pass)
    // ====================================================================
    // Run a small keyword-only search to guarantee strong keyword matches
    // survive into the candidate pool, even with small limits/overfetch.
    // Without this, exact keyword matches (e.g. unique proper nouns) get
    // buried by semantic scoring in the hybrid search.
    let keyword_first_limit = 10_i32;
    let keyword_priority_threshold: f32 = 0.8;

    let keyword_first_results = storage
        .hybrid_search_filtered(
            &args.query,
            keyword_first_limit,
            1.0, // keyword_weight = 1.0 (keyword-only)
            0.0, // semantic_weight = 0.0
            args.include_types.as_deref(),
            args.exclude_types.as_deref(),
        )
        .map_err(|e| e.to_string())?;

    // Collect keyword-priority results (keyword_score >= threshold)
    let mut keyword_priority_ids: std::collections::HashSet<String> =
        std::collections::HashSet::new();
    let mut keyword_priority_results: Vec<vestige_core::SearchResult> = Vec::new();
    for r in keyword_first_results {
        if r.keyword_score.unwrap_or(0.0) >= keyword_priority_threshold
            && r.node.retention_strength >= min_retention
        {
            keyword_priority_ids.insert(r.node.id.clone());
            keyword_priority_results.push(r);
        }
    }

    // ====================================================================
    // STAGE 1: Hybrid search with Nx over-fetch for reranking pool
    // ====================================================================
    let overfetch_multiplier = match retrieval_mode {
        "precise" => 1,    // No overfetch — return exactly what's asked
        "exhaustive" => 5, // Deep overfetch for maximum recall
        _ => 3,            // Balanced default
    };
    // When a tag_prefix OR source filter is requested, double the overfetch
    // (capped at the same 100 ceiling) so the post-filter has enough headroom
    // to still return ~limit results after thinning.
    let post_filter_multiplier = if args.tag_prefix.is_some() || source_filter.is_active() {
        2
    } else {
        1
    };
    let overfetch_limit = (limit * overfetch_multiplier * post_filter_multiplier).min(100); // Cap at 100 to avoid excessive DB load

    let results = storage
        .hybrid_search_filtered(
            &args.query,
            overfetch_limit,
            keyword_weight,
            semantic_weight,
            args.include_types.as_deref(),
            args.exclude_types.as_deref(),
        )
        .map_err(|e| e.to_string())?;

    // Filter by min_retention and min_similarity first (cheap filters)
    let mut filtered_results: Vec<_> = results
        .into_iter()
        .filter(|r| {
            if r.node.retention_strength < min_retention {
                return false;
            }
            if let Some(sem_score) = r.semantic_score
                && sem_score < min_similarity
            {
                return false;
            }
            true
        })
        .collect();

    // Apply tag_prefix post-filter BEFORE the reranker so the (expensive)
    // cross-encoder does not waste cycles on memories the caller will not
    // receive. The Stage 0 keyword-priority merge below also respects the
    // filter when applied, since merged items must have survived this step
    // OR be re-introduced from keyword_priority_results (which we re-filter).
    if let Some(prefix) = args.tag_prefix.as_deref() {
        filtered_results.retain(|r| tags_match_prefix(&r.node.tags, prefix));
    }
    // #57 Phase 4 — source-aware investigation post-filter (same precedent).
    if source_filter.is_active() {
        filtered_results.retain(|r| node_matches_source(&r.node, &source_filter));
    }

    // ====================================================================
    // Dedup: merge Stage 0 keyword-priority results into Stage 1 results
    // ====================================================================
    for kp in &keyword_priority_results {
        // Respect tag_prefix here too — Stage 0 ran without it and can
        // re-introduce filtered-out memories on the "new result" branch.
        if let Some(prefix) = args.tag_prefix.as_deref()
            && !tags_match_prefix(&kp.node.tags, prefix)
        {
            continue;
        }
        // Respect the source filter on re-inject for the same reason.
        if source_filter.is_active() && !node_matches_source(&kp.node, &source_filter) {
            continue;
        }
        if let Some(existing) = filtered_results
            .iter_mut()
            .find(|r| r.node.id == kp.node.id)
        {
            // Preserve keyword_score from Stage 0 (keyword-only search is authoritative)
            if kp.keyword_score.unwrap_or(0.0) > existing.keyword_score.unwrap_or(0.0) {
                existing.keyword_score = kp.keyword_score;
            }
            if kp.combined_score > existing.combined_score {
                existing.combined_score = kp.combined_score;
            }
        } else {
            // New result from Stage 0 not in Stage 1 — add it
            filtered_results.push(kp.clone());
        }
    }

    // ====================================================================
    // STAGE 2: Reranker (BM25-like rescoring, trim to requested limit)
    // ====================================================================
    // Keyword bypass: results with strong keyword matches (>= 0.8) skip the
    // cross-encoder entirely and are placed above reranked results. This
    // prevents the cross-encoder from burying exact/near-exact keyword hits
    // (e.g. unique proper nouns) beneath semantically-similar but unrelated
    // results.
    {
        let keyword_bypass_threshold: f32 = 0.8;
        let limit_usize = limit as usize;

        // Partition: keyword bypass vs. candidates for reranking
        let mut bypass_results: Vec<vestige_core::SearchResult> = Vec::new();
        let mut rerank_candidates: Vec<(vestige_core::SearchResult, String)> = Vec::new();

        for r in filtered_results.iter() {
            if r.keyword_score.unwrap_or(0.0) >= keyword_bypass_threshold {
                bypass_results.push(r.clone());
            } else {
                rerank_candidates.push((r.clone(), r.node.content.clone()));
            }
        }

        // Boost bypass results so they survive later pipeline stages
        // (temporal, FSRS, utility, competition) and the final re-sort.
        for r in bypass_results.iter_mut() {
            r.combined_score *= 2.0;
        }

        bypass_results.sort_by(|a, b| {
            b.combined_score
                .partial_cmp(&a.combined_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Rerank the remaining candidates when the vector-search search stack is enabled.
        #[cfg(feature = "vector-search")]
        let reranked_results: Vec<vestige_core::SearchResult> = if rerank_candidates.is_empty() {
            Vec::new()
        } else if let Ok(mut cog) = cognitive.try_lock() {
            if let Ok(reranked) =
                cog.reranker
                    .rerank(&args.query, rerank_candidates, Some(limit_usize))
            {
                reranked.into_iter().map(|rr| rr.item).collect()
            } else {
                // Reranker failed — fall back to original order for non-bypass candidates
                filtered_results
                    .iter()
                    .filter(|r| r.keyword_score.unwrap_or(0.0) < keyword_bypass_threshold)
                    .cloned()
                    .collect()
            }
        } else {
            // Couldn't acquire cognitive lock — use original order
            filtered_results
                .iter()
                .filter(|r| r.keyword_score.unwrap_or(0.0) < keyword_bypass_threshold)
                .cloned()
                .collect()
        };
        #[cfg(not(feature = "vector-search"))]
        let reranked_results: Vec<vestige_core::SearchResult> = rerank_candidates
            .into_iter()
            .map(|(result, _)| result)
            .collect();

        // Merge: bypass first, then reranked, trim to limit
        filtered_results = bypass_results;
        filtered_results.extend(reranked_results);
        filtered_results.truncate(limit_usize);
    }

    // ====================================================================
    // STAGE 3: Temporal boosting (recency + validity windows)
    // ====================================================================
    #[cfg(feature = "vector-search")]
    if let Ok(cog) = cognitive.try_lock() {
        for result in &mut filtered_results {
            let recency = cog.temporal_searcher.recency_boost(result.node.created_at);
            let validity = cog.temporal_searcher.validity_boost(
                result.node.valid_from,
                result.node.valid_until,
                None,
            );
            // Blend: 85% relevance + 15% temporal signal
            let temporal_factor = recency * validity;
            result.combined_score = result.combined_score * 0.85
                + (result.combined_score * temporal_factor as f32) * 0.15;
        }
    }

    // ====================================================================
    // STAGE 4: Memory state accessibility filtering
    // ====================================================================
    if let Ok(cog) = cognitive.try_lock() {
        for result in &mut filtered_results {
            // Build a MemoryLifecycle from node data for the calculator
            let mut lifecycle = MemoryLifecycle::new();
            lifecycle.last_access = result.node.last_accessed;
            lifecycle.access_count = result.node.reps as u32;
            // Determine state from retention strength
            lifecycle.state = if result.node.retention_strength > 0.7 {
                MemoryState::Active
            } else if result.node.retention_strength > 0.4 {
                MemoryState::Dormant
            } else if result.node.retention_strength > 0.1 {
                MemoryState::Silent
            } else {
                MemoryState::Unavailable
            };

            let mut adjusted = cog
                .accessibility_calc
                .calculate(&lifecycle, result.combined_score as f64);

            // v2.0.5: Active forgetting penalty (Anderson 2025 SIF).
            // Each prior suppress call compounds a retrieval-score penalty,
            // saturating at 80%. Applied AFTER accessibility so the penalty
            // stacks on top of any passive FSRS decay.
            if result.node.suppression_count > 0 {
                let sys =
                    vestige_core::neuroscience::active_forgetting::ActiveForgettingSystem::new();
                let penalty = sys.retrieval_penalty(result.node.suppression_count);
                adjusted *= 1.0 - penalty;
            }

            result.combined_score = adjusted as f32;
        }
    }

    // ====================================================================
    // STAGE 5: Context matching (Tulving 1973 encoding specificity)
    // ====================================================================
    if let Some(ref topics) = args.context_topics
        && !topics.is_empty()
    {
        let retrieval_ctx =
            EncodingContext::new().with_topical(TopicalContext::with_topics(topics.clone()));
        if let Ok(cog) = cognitive.try_lock() {
            for result in &mut filtered_results {
                // Build encoding context from memory's tags
                let encoding_ctx = EncodingContext::new()
                    .with_topical(TopicalContext::with_topics(result.node.tags.clone()));
                let context_score = cog
                    .context_matcher
                    .match_contexts(&encoding_ctx, &retrieval_ctx);
                // Blend: context match boosts relevance up to +30%
                result.combined_score *= 1.0 + (context_score as f32 * 0.3);
            }
        }
    }

    // Context reinstatement for top result (helps Claude understand WHY this memory matched)
    let reinstatement_info: Option<Value> = if let Ok(cog) = cognitive.try_lock() {
        if let Some(first) = filtered_results.first() {
            let current_ctx = if let Some(ref topics) = args.context_topics {
                EncodingContext::new().with_topical(TopicalContext::with_topics(topics.clone()))
            } else {
                EncodingContext::new()
            };
            let reinstatement = cog
                .context_matcher
                .reinstate_context(&first.node.id, &current_ctx);
            Some(serde_json::json!({
                "memoryId": reinstatement.memory_id,
                "temporalHint": reinstatement.temporal_hint,
                "topicalHint": reinstatement.topical_hint,
                "sessionHint": reinstatement.session_hint,
                "relatedMemories": reinstatement.related_memories,
            }))
        } else {
            None
        }
    } else {
        None
    };

    // ====================================================================
    // STAGE 5B: Retrieval competition (Anderson et al. 1994)
    // Skipped in precise mode (no need) and exhaustive mode (want all results)
    // ====================================================================
    let mut suppressed_count = 0_usize;
    if retrieval_mode == "balanced"
        && filtered_results.len() > 1
        && let Ok(mut cog) = cognitive.try_lock()
    {
        let candidates: Vec<CompetitionCandidate> = filtered_results
            .iter()
            .map(|r| CompetitionCandidate {
                memory_id: r.node.id.clone(),
                relevance_score: r.combined_score as f64,
                similarity_to_query: r.semantic_score.unwrap_or(0.0) as f64,
            })
            .collect();
        if let Some(result) = cog.competition_mgr.run_competition(&candidates, 0.7) {
            // Apply suppression: losers get penalized
            for suppressed_id in &result.suppressed_ids {
                if let Some(r) = filtered_results
                    .iter_mut()
                    .find(|r| &r.node.id == suppressed_id)
                {
                    r.combined_score *= 0.85; // 15% suppression penalty
                    suppressed_count += 1;
                }
            }
        }
    }

    // ====================================================================
    // STAGE 5C: Utility-based ranking (MemRL-inspired)
    // Memories that proved useful in past sessions get a retrieval boost.
    // utility_score = times_useful / times_retrieved (0.0 to 1.0)
    // ====================================================================
    for result in &mut filtered_results {
        let utility = result.node.utility_score.unwrap_or(0.0) as f32;
        if utility > 0.0 {
            // Utility boost: up to +15% for memories with utility_score = 1.0
            result.combined_score *= 1.0 + (utility * 0.15);
        }
    }

    // Re-sort by adjusted combined_score (descending) after all score modifications
    filtered_results.sort_by(|a, b| {
        b.combined_score
            .partial_cmp(&a.combined_score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // ====================================================================
    // STAGE 6: Spreading activation (find associated memories)
    // Skipped in precise mode. Deeper (5 results) in exhaustive mode.
    // ====================================================================
    let activation_take = match retrieval_mode {
        "precise" => 0,    // Skip entirely
        "exhaustive" => 5, // Deeper graph traversal
        _ => 3,            // Balanced default
    };
    let associations: Vec<Value> = if activation_take > 0 {
        if let Ok(mut cog) = cognitive.try_lock() {
            if let Some(first) = filtered_results.first() {
                let activated = cog.activation_network.activate(&first.node.id, 1.0);
                activated
                    .iter()
                    .take(activation_take)
                    .map(|a| {
                        serde_json::json!({
                            "memoryId": a.memory_id,
                            "activation": a.activation,
                            "distance": a.distance,
                        })
                    })
                    .collect()
            } else {
                vec![]
            }
        } else {
            vec![]
        }
    } else {
        vec![]
    };

    // ====================================================================
    // Auto-strengthen on access (Testing Effect)
    // ====================================================================
    let ids: Vec<&str> = filtered_results
        .iter()
        .map(|r| r.node.id.as_str())
        .collect();
    let _ = storage.strengthen_batch_on_access(&ids);

    // Drop storage lock before acquiring cognitive for side effects

    // ====================================================================
    // STAGE 7: Side effects — predictive memory + reconsolidation
    // ====================================================================
    if let Ok(mut cog) = cognitive.try_lock() {
        // 7A. Record query for predictive memory
        let _ = cog.predictive_memory.record_query(&args.query, &[]);

        // 7B. Record each accessed memory for predictive/speculative models
        for result in &filtered_results {
            let _ = cog.predictive_memory.record_memory_access(
                &result.node.id,
                &result.node.content.chars().take(100).collect::<String>(),
                &result.node.tags,
            );

            cog.speculative_retriever.record_access(
                &result.node.id,
                None,                      // file_context
                Some(args.query.as_str()), // query_context
                None,                      // was_helpful (unknown yet)
            );

            // 7C. Mark labile for reconsolidation window (5 min)
            let snapshot = MemorySnapshot {
                content: result.node.content.clone(),
                tags: result.node.tags.clone(),
                retention_strength: result.node.retention_strength,
                storage_strength: result.node.storage_strength,
                retrieval_strength: result.node.retrieval_strength,
                connection_ids: vec![],
                captured_at: Utc::now(),
            };
            cog.reconsolidation.mark_labile(&result.node.id, snapshot);
        }

        // 7D. Feed the co-access channel: the memories returned together by one
        // query WERE accessed together, so record_usage learns their co-access
        // patterns (trigger -> predicted). Without this call the speculative
        // retriever's co-access prediction channel stays permanently empty.
        if filtered_results.len() >= 2 {
            let accessed: Vec<String> =
                filtered_results.iter().map(|r| r.node.id.clone()).collect();
            cog.speculative_retriever.record_usage(&[], &accessed);
        }
    }

    // ====================================================================
    // Format and return
    // ====================================================================
    let mut formatted: Vec<Value> = filtered_results
        .iter()
        .map(|r| format_search_result(r, detail_level))
        .collect();
    apply_output_masks(&mut formatted, output_config);

    // ====================================================================
    // Token budget enforcement (v1.8.0)
    // ====================================================================
    let mut budget_expandable: Vec<String> = Vec::new();
    let mut budget_tokens_used: Option<usize> = None;
    if let Some(budget) = args.token_budget {
        let budget = budget.clamp(100, 100000) as usize;
        let budget_chars = budget * 4;
        let mut used = 0;
        let mut budgeted = Vec::new();

        for result in &formatted {
            let size = serde_json::to_string(result).unwrap_or_default().len();
            if used + size > budget_chars {
                if let Some(id) = result.get("id").and_then(|v| v.as_str()) {
                    budget_expandable.push(id.to_string());
                }
                continue;
            }
            used += size;
            budgeted.push(result.clone());
        }

        budget_tokens_used = Some(used / 4);
        formatted = budgeted;
    }

    // Check learning mode via attention signal
    let learning_mode = cognitive
        .try_lock()
        .ok()
        .map(|cog| cog.attention_signal.is_learning_mode())
        .unwrap_or(false);

    let mut response = serde_json::json!({
        "query": args.query,
        "method": "hybrid+cognitive",
        "retrievalMode": retrieval_mode,
        "detailLevel": detail_level,
        "profile": output_config.profile.as_str(),
        "total": formatted.len(),
        "results": formatted,
    });

    // Helpful hint when no results found
    if formatted.is_empty() {
        response["hint"] = serde_json::json!(
            "No memories found. Use smart_ingest to add memories, or try a broader query."
        );
    }

    // Include associations if any were found
    if !associations.is_empty() {
        response["associations"] = serde_json::json!(associations);
    }
    // Include context reinstatement if computed
    if let Some(ri) = reinstatement_info {
        response["contextReinstatement"] = ri;
    }
    // Include competition stats
    if suppressed_count > 0 {
        response["competitionSuppressed"] = serde_json::json!(suppressed_count);
    }
    // Include learning mode detection
    if learning_mode {
        response["learningModeDetected"] = serde_json::json!(true);
    }
    // Include token budget info (v1.8.0)
    if !budget_expandable.is_empty() {
        response["expandable"] = serde_json::json!(budget_expandable);
    }
    if let Some(budget) = args.token_budget {
        response["tokenBudget"] = serde_json::json!(budget);
    }
    if let Some(used) = budget_tokens_used {
        response["tokensUsed"] = serde_json::json!(used);
    }

    Ok(response)
}

fn is_literal_query(query: &str) -> bool {
    let trimmed = query.trim();
    if trimmed.len() >= 2 {
        let bytes = trimmed.as_bytes();
        if (bytes[0] == b'"' && bytes[bytes.len() - 1] == b'"')
            || (bytes[0] == b'\'' && bytes[bytes.len() - 1] == b'\'')
        {
            return true;
        }
    }

    if uuid::Uuid::parse_str(trimmed).is_ok() {
        return true;
    }

    let token_count = trimmed.split_whitespace().count();
    if token_count != 1 {
        return false;
    }

    let has_identifier_punctuation = trimmed
        .chars()
        .any(|c| matches!(c, '_' | '-' | '/' | '\\' | '.' | ':' | '=' | '@'));
    if has_identifier_punctuation {
        return true;
    }

    let has_alpha = trimmed.chars().any(|c| c.is_ascii_alphabetic());
    has_alpha
        && trimmed.contains('_')
        && trimmed
            .chars()
            .all(|c| c.is_ascii_uppercase() || c.is_ascii_digit() || c == '_')
}

/// Returns `true` when the given tag list contains at least one tag whose
/// string value starts with `prefix`. Empty prefix matches every result with
/// at least one tag (and never matches a tagless result).
///
/// Case-sensitive by design: the existing tag-match semantics in
/// `memory_timeline` / `export` / `gc` are exact-match (case-sensitive), so
/// keeping this consistent avoids surprise. Operators wanting case-insensitive
/// prefix-search should normalize tags at ingest time.
fn tags_match_prefix(tags: &[String], prefix: &str) -> bool {
    tags.iter().any(|t| t.starts_with(prefix))
}

/// Validity filter for source-aware search (#57 Phase 4).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
enum SourceStatus {
    /// No validity constraint.
    #[default]
    Any,
    /// Only currently-valid records.
    Valid,
    /// Only tombstoned records (no longer visible upstream, kept for audit).
    Tombstoned,
}

/// Parsed source-aware investigation filter (#57 Phase 4).
///
/// All fields are optional; an all-empty filter matches every node (so search
/// behavior is byte-for-byte unchanged when no source filter is supplied). Any
/// source-scoped field being set excludes legacy/agent memories that have no
/// `source_envelope`. Applied as a post-filter on the recalled nodes, mirroring
/// the existing `tag_prefix` precedent (no SQL changes).
#[derive(Debug, Clone, Default)]
struct SourceFilter {
    system: Option<String>,
    project: Option<String>,
    id: Option<String>,
    source_type: Option<String>,
    author: Option<String>,
    updated_after: Option<chrono::DateTime<chrono::Utc>>,
    updated_before: Option<chrono::DateTime<chrono::Utc>>,
    status: SourceStatus,
}

impl SourceFilter {
    /// Build from raw args, hard-erroring on malformed timestamps / status enum
    /// (consistent with how `detail_level` / `retrieval_mode` reject bad input —
    /// a silently-`None` bound would widen the filter and return wrong rows).
    fn from_args(args: &SearchArgs) -> Result<Self, String> {
        let parse_ts = |s: &Option<String>,
                        field: &str|
         -> Result<Option<chrono::DateTime<chrono::Utc>>, String> {
            match s {
                None => Ok(None),
                Some(v) => chrono::DateTime::parse_from_rfc3339(v)
                    .map(|dt| Some(dt.with_timezone(&chrono::Utc)))
                    .map_err(|_| format!("Invalid {field}: '{v}' is not an RFC3339 timestamp")),
            }
        };
        let status = match args.source_status.as_deref() {
            None | Some("any") => SourceStatus::Any,
            Some("valid") => SourceStatus::Valid,
            Some("tombstoned") => SourceStatus::Tombstoned,
            Some(other) => {
                return Err(format!(
                    "Invalid source_status '{other}'. Must be 'any', 'valid', or 'tombstoned'."
                ));
            }
        };
        Ok(Self {
            system: args.source_system.clone(),
            project: args.source_project.clone(),
            id: args.source_id.clone(),
            source_type: args.source_type.clone(),
            author: args.source_author.clone(),
            updated_after: parse_ts(&args.source_updated_after, "source_updated_after")?,
            updated_before: parse_ts(&args.source_updated_before, "source_updated_before")?,
            status,
        })
    }

    /// True when at least one filter is set (used to size the over-fetch pool).
    fn is_active(&self) -> bool {
        self.system.is_some()
            || self.project.is_some()
            || self.id.is_some()
            || self.source_type.is_some()
            || self.author.is_some()
            || self.updated_after.is_some()
            || self.updated_before.is_some()
            || self.status != SourceStatus::Any
    }
}

/// Predicate: does this node satisfy the source-aware investigation filter?
/// An all-empty filter returns `true` for every node.
fn node_matches_source(node: &vestige_core::KnowledgeNode, filter: &SourceFilter) -> bool {
    // Validity check operates on the NODE (valid_until lives on the node).
    match filter.status {
        SourceStatus::Any => {}
        SourceStatus::Valid if !node.is_currently_valid() => return false,
        SourceStatus::Tombstoned if node.is_currently_valid() => return false,
        _ => {}
    }

    // Any source-scoped field requires an envelope; legacy memories are out.
    // This includes `source_status=valid`: otherwise a source-scoped query for
    // valid connector records would also return ordinary valid agent memories.
    let envelope_scoped = filter.system.is_some()
        || filter.project.is_some()
        || filter.id.is_some()
        || filter.source_type.is_some()
        || filter.author.is_some()
        || filter.updated_after.is_some()
        || filter.updated_before.is_some()
        || filter.status != SourceStatus::Any;
    if !envelope_scoped {
        return true;
    }
    let Some(env) = node.source_envelope.as_ref() else {
        return false;
    };

    let exact = |want: &Option<String>, have: &Option<String>| -> bool {
        match want {
            None => true,
            Some(w) => have.as_deref() == Some(w.as_str()),
        }
    };
    if !exact(&filter.system, &env.source_system) {
        return false;
    }
    if !exact(&filter.project, &env.source_project) {
        return false;
    }
    if !exact(&filter.id, &env.source_id) {
        return false;
    }
    if !exact(&filter.source_type, &env.source_type) {
        return false;
    }
    if !exact(&filter.author, &env.source_author) {
        return false;
    }
    // Date bounds (inclusive) on the source-updated time.
    if filter.updated_after.is_some() || filter.updated_before.is_some() {
        let Some(ts) = env.source_updated_at else {
            return false;
        };
        if let Some(after) = filter.updated_after
            && ts < after
        {
            return false;
        }
        if let Some(before) = filter.updated_before
            && ts > before
        {
            return false;
        }
    }
    true
}

/// Format a search result based on the requested detail level.
/// Score field keys dropped when an output profile suppresses scores.
const SCORE_FIELDS: &[&str] = &["combinedScore", "keywordScore", "semanticScore"];
/// Timestamp field keys dropped when an output profile suppresses timestamps.
const TIMESTAMP_FIELDS: &[&str] = &[
    "createdAt",
    "updatedAt",
    "lastAccessed",
    "nextReview",
    "validFrom",
    "validUntil",
];

/// Strip score/timestamp fields from already-formatted result objects according
/// to the active output profile (e.g. the `lean` profile drops both). Tools
/// call this after formatting so the field-mask behavior is centralized and the
/// per-detail-level formatters stay unchanged.
pub fn apply_output_masks(results: &mut [Value], output_config: &OutputConfig) {
    if output_config.show_scores && output_config.show_timestamps {
        return;
    }
    for result in results.iter_mut() {
        if let Some(obj) = result.as_object_mut() {
            if !output_config.show_scores {
                for key in SCORE_FIELDS {
                    obj.remove(*key);
                }
            }
            if !output_config.show_timestamps {
                for key in TIMESTAMP_FIELDS {
                    obj.remove(*key);
                }
            }
        }
    }
}

/// Build a compact `source` object from a node's connector provenance (#57),
/// or `Value::Null` when the memory has no external source envelope.
///
/// Surfacing `url` in search results is the whole point of the connector layer:
/// the agent can follow the citation back to the canonical Redmine/GitHub record
/// for authoritative detail. `tombstoned` flags records no longer visible
/// upstream (kept for audit).
fn source_provenance(node: &vestige_core::KnowledgeNode) -> Value {
    let Some(env) = node.source_envelope.as_ref() else {
        return Value::Null;
    };
    serde_json::json!({
        "system": env.source_system,
        "id": env.source_id,
        "url": env.source_url,
        "project": env.source_project,
        "type": env.source_type,
        "author": env.source_author,
        "sourceUpdatedAt": env.source_updated_at.map(|dt| dt.to_rfc3339()),
        "syncedAt": env.synced_at.map(|dt| dt.to_rfc3339()),
        // A tombstoned (no-longer-visible) record has valid_until set in the past.
        "tombstoned": !node.is_currently_valid(),
    })
}

fn format_search_result(r: &vestige_core::SearchResult, detail_level: &str) -> Value {
    match detail_level {
        "brief" => serde_json::json!({
            "id": r.node.id,
            "nodeType": r.node.node_type,
            "tags": r.node.tags,
            "retentionStrength": r.node.retention_strength,
            "combinedScore": r.combined_score,
        }),
        "full" => {
            let mut v = serde_json::json!({
                "id": r.node.id,
                "content": r.node.content,
                "combinedScore": r.combined_score,
                "keywordScore": r.keyword_score,
                "semanticScore": r.semantic_score,
                "nodeType": r.node.node_type,
                "tags": r.node.tags,
                "retentionStrength": r.node.retention_strength,
                "storageStrength": r.node.storage_strength,
                "retrievalStrength": r.node.retrieval_strength,
                "source": r.node.source,
                "sentimentScore": r.node.sentiment_score,
                "sentimentMagnitude": r.node.sentiment_magnitude,
                "createdAt": r.node.created_at.to_rfc3339(),
                "updatedAt": r.node.updated_at.to_rfc3339(),
                "lastAccessed": r.node.last_accessed.to_rfc3339(),
                "nextReview": r.node.next_review.map(|dt| dt.to_rfc3339()),
                "stability": r.node.stability,
                "difficulty": r.node.difficulty,
                "reps": r.node.reps,
                "lapses": r.node.lapses,
                "validFrom": r.node.valid_from.map(|dt| dt.to_rfc3339()),
                "validUntil": r.node.valid_until.map(|dt| dt.to_rfc3339()),
                "matchType": format!("{:?}", r.match_type),
            });
            attach_source_record(&mut v, &r.node);
            v
        }
        // "summary" (default) — includes dates so AI never has to guess when a memory is from
        _ => {
            let mut v = serde_json::json!({
                "id": r.node.id,
                "content": r.node.content,
                "combinedScore": r.combined_score,
                "keywordScore": r.keyword_score,
                "semanticScore": r.semantic_score,
                "nodeType": r.node.node_type,
                "tags": r.node.tags,
                "retentionStrength": r.node.retention_strength,
                "createdAt": r.node.created_at.to_rfc3339(),
                "updatedAt": r.node.updated_at.to_rfc3339(),
            });
            attach_source_record(&mut v, &r.node);
            v
        }
    }
}

/// Inject a `sourceRecord` object into a result `Value` ONLY when the memory
/// has external provenance, so legacy (agent/user-authored) memories keep their
/// exact prior result shape.
fn attach_source_record(value: &mut Value, node: &vestige_core::KnowledgeNode) {
    let provenance = source_provenance(node);
    if !provenance.is_null()
        && let Value::Object(map) = value
    {
        map.insert("sourceRecord".to_string(), provenance);
    }
}

/// Format a KnowledgeNode based on the requested detail level.
/// Reusable across search, timeline, and other tools.
pub fn format_node(node: &vestige_core::KnowledgeNode, detail_level: &str) -> Value {
    match detail_level {
        "brief" => serde_json::json!({
            "id": node.id,
            "nodeType": node.node_type,
            "tags": node.tags,
            "retentionStrength": node.retention_strength,
        }),
        "full" => {
            let mut v = serde_json::json!({
                "id": node.id,
                "content": node.content,
                "nodeType": node.node_type,
                "tags": node.tags,
                "retentionStrength": node.retention_strength,
                "storageStrength": node.storage_strength,
                "retrievalStrength": node.retrieval_strength,
                "source": node.source,
                "sentimentScore": node.sentiment_score,
                "sentimentMagnitude": node.sentiment_magnitude,
                "createdAt": node.created_at.to_rfc3339(),
                "updatedAt": node.updated_at.to_rfc3339(),
                "lastAccessed": node.last_accessed.to_rfc3339(),
                "nextReview": node.next_review.map(|dt| dt.to_rfc3339()),
                "stability": node.stability,
                "difficulty": node.difficulty,
                "reps": node.reps,
                "lapses": node.lapses,
                "validFrom": node.valid_from.map(|dt| dt.to_rfc3339()),
                "validUntil": node.valid_until.map(|dt| dt.to_rfc3339()),
            });
            attach_source_record(&mut v, node);
            v
        }
        // "summary" (default)
        _ => {
            let mut v = serde_json::json!({
                "id": node.id,
                "content": node.content,
                "nodeType": node.node_type,
                "tags": node.tags,
                "retentionStrength": node.retention_strength,
            });
            attach_source_record(&mut v, node);
            v
        }
    }
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cognitive::CognitiveEngine;
    use tempfile::TempDir;
    use vestige_core::IngestInput;

    fn test_cognitive() -> Arc<Mutex<CognitiveEngine>> {
        Arc::new(Mutex::new(CognitiveEngine::new()))
    }

    /// Create a test storage instance with a temporary database
    async fn test_storage() -> (Arc<Storage>, TempDir) {
        let dir = TempDir::new().unwrap();
        let storage = Storage::new(Some(dir.path().join("test.db"))).unwrap();
        (Arc::new(storage), dir)
    }

    /// Helper to ingest test content
    async fn ingest_test_content(storage: &Arc<Storage>, content: &str) -> String {
        let input = IngestInput {
            content: content.to_string(),
            node_type: "fact".to_string(),
            source: None,
            sentiment_score: 0.0,
            sentiment_magnitude: 0.0,
            tags: vec![],
            valid_from: None,
            valid_until: None,
            source_envelope: None,
        };
        let node = storage.ingest(input).unwrap();
        node.id
    }

    // ========================================================================
    // QUERY VALIDATION TESTS
    // ========================================================================

    #[tokio::test]
    async fn test_search_empty_query_fails() {
        let (storage, _dir) = test_storage().await;
        let args = serde_json::json!({ "query": "" });
        let result = execute(
            &storage,
            &test_cognitive(),
            &OutputConfig::default(),
            Some(args),
        )
        .await;
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("empty"));
    }

    #[tokio::test]
    async fn test_search_whitespace_only_query_fails() {
        let (storage, _dir) = test_storage().await;
        let args = serde_json::json!({ "query": "   \t\n  " });
        let result = execute(
            &storage,
            &test_cognitive(),
            &OutputConfig::default(),
            Some(args),
        )
        .await;
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("empty"));
    }

    #[tokio::test]
    async fn test_search_missing_arguments_fails() {
        let (storage, _dir) = test_storage().await;
        let result = execute(&storage, &test_cognitive(), &OutputConfig::default(), None).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Missing arguments"));
    }

    #[tokio::test]
    async fn test_search_missing_query_field_fails() {
        let (storage, _dir) = test_storage().await;
        let args = serde_json::json!({ "limit": 10 });
        let result = execute(
            &storage,
            &test_cognitive(),
            &OutputConfig::default(),
            Some(args),
        )
        .await;
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Invalid arguments"));
    }

    #[test]
    fn test_literal_query_detection() {
        assert!(is_literal_query("\"exact phrase\""));
        assert!(is_literal_query("OPENAI_API_KEY"));
        assert!(is_literal_query("mlx_lm.server"));
        assert!(is_literal_query("src/main.rs"));
        assert!(is_literal_query("4da778e2-1111-4444-8888-123456789abc"));
        assert!(!is_literal_query("how should memory search work"));
    }

    #[tokio::test]
    async fn test_concrete_search_env_var_lands_first() {
        let (storage, _dir) = test_storage().await;
        ingest_test_content(
            &storage,
            "General OpenAI setup and API key rotation guidance",
        )
        .await;
        let target = ingest_test_content(
            &storage,
            "Release smoke test requires OPENAI_API_KEY to be set in the shell",
        )
        .await;
        ingest_test_content(
            &storage,
            "Credentials should be stored outside the repository",
        )
        .await;

        let args = serde_json::json!({
            "query": "OPENAI_API_KEY",
            "limit": 5
        });
        let result = execute(
            &storage,
            &test_cognitive(),
            &OutputConfig::default(),
            Some(args),
        )
        .await
        .unwrap();

        assert_eq!(result["method"], "concrete");
        assert_eq!(result["concrete"], true);
        assert_eq!(result["results"][0]["id"], target);
    }

    #[tokio::test]
    async fn test_concrete_search_uuid_lands_first() {
        let (storage, _dir) = test_storage().await;
        let uuid = "4da778e2-1111-4444-8888-123456789abc";
        ingest_test_content(&storage, "Several memories mention release identifiers").await;
        let target = ingest_test_content(
            &storage,
            &format!("The migration bug is tracked by exact id {}", uuid),
        )
        .await;

        let args = serde_json::json!({
            "query": uuid,
            "limit": 5
        });
        let result = execute(
            &storage,
            &test_cognitive(),
            &OutputConfig::default(),
            Some(args),
        )
        .await
        .unwrap();

        assert_eq!(result["method"], "concrete");
        assert_eq!(result["results"][0]["id"], target);
    }

    #[tokio::test]
    async fn test_concrete_search_process_name_lands_first() {
        let (storage, _dir) = test_storage().await;
        ingest_test_content(
            &storage,
            "The local MLX server can expose an OpenAI-compatible endpoint",
        )
        .await;
        let target = ingest_test_content(
            &storage,
            "If mlx_lm.server is already running, do not start a second Sanhedrin backend",
        )
        .await;

        let args = serde_json::json!({
            "query": "mlx_lm.server",
            "limit": 5
        });
        let result = execute(
            &storage,
            &test_cognitive(),
            &OutputConfig::default(),
            Some(args),
        )
        .await
        .unwrap();

        assert_eq!(result["method"], "concrete");
        assert_eq!(result["results"][0]["id"], target);
    }

    // ========================================================================
    // LIMIT CLAMPING TESTS
    // ========================================================================

    #[tokio::test]
    async fn test_search_limit_clamped_to_minimum() {
        let (storage, _dir) = test_storage().await;
        ingest_test_content(&storage, "Test content for limit clamping").await;

        // Try with limit 0 - should clamp to 1
        let args = serde_json::json!({
            "query": "test",
            "limit": 0
        });
        let result = execute(
            &storage,
            &test_cognitive(),
            &OutputConfig::default(),
            Some(args),
        )
        .await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_search_limit_clamped_to_maximum() {
        let (storage, _dir) = test_storage().await;
        ingest_test_content(&storage, "Test content for max limit").await;

        // Try with limit 1000 - should clamp to 100
        let args = serde_json::json!({
            "query": "test",
            "limit": 1000
        });
        let result = execute(
            &storage,
            &test_cognitive(),
            &OutputConfig::default(),
            Some(args),
        )
        .await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_search_negative_limit_clamped() {
        let (storage, _dir) = test_storage().await;
        ingest_test_content(&storage, "Test content for negative limit").await;

        let args = serde_json::json!({
            "query": "test",
            "limit": -5
        });
        let result = execute(
            &storage,
            &test_cognitive(),
            &OutputConfig::default(),
            Some(args),
        )
        .await;
        assert!(result.is_ok());
    }

    // ========================================================================
    // MIN_RETENTION CLAMPING TESTS
    // ========================================================================

    #[tokio::test]
    async fn test_search_min_retention_clamped_to_zero() {
        let (storage, _dir) = test_storage().await;
        ingest_test_content(&storage, "Test content for retention clamping").await;

        let args = serde_json::json!({
            "query": "test",
            "min_retention": -0.5
        });
        let result = execute(
            &storage,
            &test_cognitive(),
            &OutputConfig::default(),
            Some(args),
        )
        .await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_search_min_retention_clamped_to_one() {
        let (storage, _dir) = test_storage().await;
        ingest_test_content(&storage, "Test content for max retention").await;

        let args = serde_json::json!({
            "query": "test",
            "min_retention": 1.5
        });
        let result = execute(
            &storage,
            &test_cognitive(),
            &OutputConfig::default(),
            Some(args),
        )
        .await;
        // Should succeed but may return no results (retention > 1.0 clamped to 1.0)
        assert!(result.is_ok());
    }

    // ========================================================================
    // MIN_SIMILARITY CLAMPING TESTS
    // ========================================================================

    #[tokio::test]
    async fn test_search_min_similarity_clamped_to_zero() {
        let (storage, _dir) = test_storage().await;
        ingest_test_content(&storage, "Test content for similarity clamping").await;

        let args = serde_json::json!({
            "query": "test",
            "min_similarity": -0.5
        });
        let result = execute(
            &storage,
            &test_cognitive(),
            &OutputConfig::default(),
            Some(args),
        )
        .await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_search_min_similarity_clamped_to_one() {
        let (storage, _dir) = test_storage().await;
        ingest_test_content(&storage, "Test content for max similarity").await;

        let args = serde_json::json!({
            "query": "test",
            "min_similarity": 1.5
        });
        let result = execute(
            &storage,
            &test_cognitive(),
            &OutputConfig::default(),
            Some(args),
        )
        .await;
        // Should succeed but may return no results
        assert!(result.is_ok());
    }

    // ========================================================================
    // SUCCESSFUL SEARCH TESTS
    // ========================================================================

    #[tokio::test]
    async fn test_search_basic_query_succeeds() {
        let (storage, _dir) = test_storage().await;
        ingest_test_content(&storage, "The Rust programming language is memory safe.").await;

        let args = serde_json::json!({ "query": "rust" });
        let result = execute(
            &storage,
            &test_cognitive(),
            &OutputConfig::default(),
            Some(args),
        )
        .await;
        assert!(result.is_ok());

        let value = result.unwrap();
        assert_eq!(value["query"], "rust");
        assert_eq!(value["method"], "hybrid+cognitive");
        assert!(value["total"].is_number());
        assert!(value["results"].is_array());
    }

    #[tokio::test]
    async fn test_search_returns_matching_content() {
        let (storage, _dir) = test_storage().await;
        let node_id =
            ingest_test_content(&storage, "Python is a dynamic programming language.").await;

        let args = serde_json::json!({
            "query": "python",
            "min_similarity": 0.0
        });
        let result = execute(
            &storage,
            &test_cognitive(),
            &OutputConfig::default(),
            Some(args),
        )
        .await;
        assert!(result.is_ok());

        let value = result.unwrap();
        let results = value["results"].as_array().unwrap();
        assert!(!results.is_empty());
        assert_eq!(results[0]["id"], node_id);
    }

    #[tokio::test]
    async fn test_search_with_limit() {
        let (storage, _dir) = test_storage().await;
        // Ingest multiple items
        ingest_test_content(&storage, "Testing content one").await;
        ingest_test_content(&storage, "Testing content two").await;
        ingest_test_content(&storage, "Testing content three").await;

        let args = serde_json::json!({
            "query": "testing",
            "limit": 2,
            "min_similarity": 0.0
        });
        let result = execute(
            &storage,
            &test_cognitive(),
            &OutputConfig::default(),
            Some(args),
        )
        .await;
        assert!(result.is_ok());

        let value = result.unwrap();
        let results = value["results"].as_array().unwrap();
        assert!(results.len() <= 2);
    }

    #[tokio::test]
    async fn test_search_empty_database_returns_empty_array() {
        let (storage, _dir) = test_storage().await;
        // Don't ingest anything - database is empty

        let args = serde_json::json!({ "query": "anything" });
        let result = execute(
            &storage,
            &test_cognitive(),
            &OutputConfig::default(),
            Some(args),
        )
        .await;
        assert!(result.is_ok());

        let value = result.unwrap();
        assert_eq!(value["total"], 0);
        assert!(value["results"].as_array().unwrap().is_empty());
    }

    #[tokio::test]
    async fn test_search_result_contains_expected_fields() {
        let (storage, _dir) = test_storage().await;
        ingest_test_content(&storage, "Testing field presence in search results.").await;

        let args = serde_json::json!({
            "query": "testing",
            "min_similarity": 0.0
        });
        let result = execute(
            &storage,
            &test_cognitive(),
            &OutputConfig::default(),
            Some(args),
        )
        .await;
        assert!(result.is_ok());

        let value = result.unwrap();
        let results = value["results"].as_array().unwrap();
        if !results.is_empty() {
            let first = &results[0];
            assert!(first["id"].is_string());
            assert!(first["content"].is_string());
            assert!(first["combinedScore"].is_number());
            // keywordScore and semanticScore may be null if not matched
            assert!(first["nodeType"].is_string());
            assert!(first["tags"].is_array());
            assert!(first["retentionStrength"].is_number());
        }
    }

    // ========================================================================
    // DEFAULT VALUES TESTS
    // ========================================================================

    #[tokio::test]
    async fn test_search_default_limit_is_10() {
        let (storage, _dir) = test_storage().await;
        // Ingest more than 10 items
        for i in 0..15 {
            ingest_test_content(&storage, &format!("Item number {}", i)).await;
        }

        let args = serde_json::json!({
            "query": "item",
            "min_similarity": 0.0
        });
        let result = execute(
            &storage,
            &test_cognitive(),
            &OutputConfig::default(),
            Some(args),
        )
        .await;
        assert!(result.is_ok());

        let value = result.unwrap();
        let results = value["results"].as_array().unwrap();
        assert!(results.len() <= 10);
    }

    // ========================================================================
    // SCHEMA TESTS
    // ========================================================================

    #[test]
    fn test_schema_has_required_fields() {
        let schema_value = schema();
        assert_eq!(schema_value["type"], "object");
        assert!(schema_value["properties"]["query"].is_object());
        assert!(
            schema_value["required"]
                .as_array()
                .unwrap()
                .contains(&serde_json::json!("query"))
        );
    }

    #[test]
    fn test_schema_has_optional_fields() {
        let schema_value = schema();
        assert!(schema_value["properties"]["limit"].is_object());
        assert!(schema_value["properties"]["min_retention"].is_object());
        assert!(schema_value["properties"]["min_similarity"].is_object());
    }

    #[test]
    fn test_schema_limit_has_bounds() {
        let schema_value = schema();
        let limit_schema = &schema_value["properties"]["limit"];
        assert_eq!(limit_schema["minimum"], 1);
        assert_eq!(limit_schema["maximum"], 100);
        assert_eq!(limit_schema["default"], 10);
    }

    #[test]
    fn test_schema_min_retention_has_bounds() {
        let schema_value = schema();
        let retention_schema = &schema_value["properties"]["min_retention"];
        assert_eq!(retention_schema["minimum"], 0.0);
        assert_eq!(retention_schema["maximum"], 1.0);
        assert_eq!(retention_schema["default"], 0.0);
    }

    #[test]
    fn test_schema_min_similarity_has_bounds() {
        let schema_value = schema();
        let similarity_schema = &schema_value["properties"]["min_similarity"];
        assert_eq!(similarity_schema["minimum"], 0.0);
        assert_eq!(similarity_schema["maximum"], 1.0);
        assert_eq!(similarity_schema["default"], 0.5);
    }

    // ========================================================================
    // DETAIL LEVEL TESTS
    // ========================================================================

    #[test]
    fn test_schema_has_detail_level() {
        let schema_value = schema();
        let dl = &schema_value["properties"]["detail_level"];
        assert!(dl.is_object());
        assert_eq!(dl["default"], "summary");
        let enum_values = dl["enum"].as_array().unwrap();
        assert!(enum_values.contains(&serde_json::json!("brief")));
        assert!(enum_values.contains(&serde_json::json!("summary")));
        assert!(enum_values.contains(&serde_json::json!("full")));
    }

    #[tokio::test]
    async fn test_search_detail_level_brief_excludes_content() {
        let (storage, _dir) = test_storage().await;
        ingest_test_content(&storage, "Brief mode test content for search.").await;

        let args = serde_json::json!({
            "query": "brief",
            "detail_level": "brief",
            "min_similarity": 0.0
        });
        let result = execute(
            &storage,
            &test_cognitive(),
            &OutputConfig::default(),
            Some(args),
        )
        .await;
        assert!(result.is_ok());

        let value = result.unwrap();
        assert_eq!(value["detailLevel"], "brief");
        let results = value["results"].as_array().unwrap();
        if !results.is_empty() {
            let first = &results[0];
            // Brief should NOT have content
            assert!(first.get("content").is_none() || first["content"].is_null());
            // Brief should have these fields
            assert!(first["id"].is_string());
            assert!(first["nodeType"].is_string());
            assert!(first["tags"].is_array());
            assert!(first["retentionStrength"].is_number());
            assert!(first["combinedScore"].is_number());
        }
    }

    #[tokio::test]
    async fn test_search_detail_level_full_includes_timestamps() {
        let (storage, _dir) = test_storage().await;
        ingest_test_content(&storage, "Full mode test content for search.").await;

        let args = serde_json::json!({
            "query": "full",
            "detail_level": "full",
            "min_similarity": 0.0
        });
        let result = execute(
            &storage,
            &test_cognitive(),
            &OutputConfig::default(),
            Some(args),
        )
        .await;
        assert!(result.is_ok());

        let value = result.unwrap();
        assert_eq!(value["detailLevel"], "full");
        let results = value["results"].as_array().unwrap();
        if !results.is_empty() {
            let first = &results[0];
            // Full should have timestamps
            assert!(first["createdAt"].is_string());
            assert!(first["updatedAt"].is_string());
            assert!(first["content"].is_string());
            assert!(first["storageStrength"].is_number());
            assert!(first["retrievalStrength"].is_number());
            assert!(first["matchType"].is_string());
        }
    }

    #[tokio::test]
    async fn test_search_detail_level_default_is_summary() {
        let (storage, _dir) = test_storage().await;
        ingest_test_content(&storage, "Default detail level test content.").await;

        let args = serde_json::json!({
            "query": "default",
            "min_similarity": 0.0
        });
        let result = execute(
            &storage,
            &test_cognitive(),
            &OutputConfig::default(),
            Some(args),
        )
        .await;
        assert!(result.is_ok());

        let value = result.unwrap();
        assert_eq!(value["detailLevel"], "summary");
        let results = value["results"].as_array().unwrap();
        if !results.is_empty() {
            let first = &results[0];
            // Summary should have content AND timestamps (v2.1: dates always visible)
            assert!(first["content"].is_string());
            assert!(first["id"].is_string());
            assert!(
                first["createdAt"].is_string(),
                "summary must include createdAt"
            );
            assert!(
                first["updatedAt"].is_string(),
                "summary must include updatedAt"
            );
        }
    }

    #[tokio::test]
    async fn test_search_detail_level_invalid_fails() {
        let (storage, _dir) = test_storage().await;
        let args = serde_json::json!({
            "query": "test",
            "detail_level": "invalid_level"
        });
        let result = execute(
            &storage,
            &test_cognitive(),
            &OutputConfig::default(),
            Some(args),
        )
        .await;
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Invalid detail_level"));
    }

    // ========================================================================
    // TOKEN BUDGET TESTS (v1.8.0)
    // ========================================================================

    #[tokio::test]
    async fn test_token_budget_limits_results() {
        let (storage, _dir) = test_storage().await;
        for i in 0..10 {
            ingest_test_content(
                &storage,
                &format!(
                    "Budget test content number {} with some extra text to increase size.",
                    i
                ),
            )
            .await;
        }

        // Small budget should reduce results
        let args = serde_json::json!({
            "query": "budget test",
            "token_budget": 200,
            "min_similarity": 0.0
        });
        let result = execute(
            &storage,
            &test_cognitive(),
            &OutputConfig::default(),
            Some(args),
        )
        .await;
        assert!(result.is_ok());

        let value = result.unwrap();
        assert!(value["tokenBudget"].as_i64().unwrap() == 200);
        assert!(value["tokensUsed"].is_number());
    }

    #[tokio::test]
    async fn test_token_budget_expandable() {
        let (storage, _dir) = test_storage().await;
        for i in 0..15 {
            ingest_test_content(
                &storage,
                &format!(
                    "Expandable budget test number {} with quite a bit of content to ensure we exceed the token budget allocation threshold.",
                    i
                ),
            )
            .await;
        }

        let args = serde_json::json!({
            "query": "expandable budget test",
            "token_budget": 150,
            "min_similarity": 0.0
        });
        let result = execute(
            &storage,
            &test_cognitive(),
            &OutputConfig::default(),
            Some(args),
        )
        .await;
        assert!(result.is_ok());

        let value = result.unwrap();
        // expandable field should exist if results were dropped
        if let Some(expandable) = value.get("expandable") {
            assert!(expandable.is_array());
        }
    }

    #[tokio::test]
    async fn test_no_budget_unchanged() {
        let (storage, _dir) = test_storage().await;
        ingest_test_content(&storage, "No budget test content.").await;

        let args = serde_json::json!({
            "query": "no budget",
            "min_similarity": 0.0
        });
        let result = execute(
            &storage,
            &test_cognitive(),
            &OutputConfig::default(),
            Some(args),
        )
        .await;
        assert!(result.is_ok());

        let value = result.unwrap();
        // No budget fields should be present
        assert!(value.get("tokenBudget").is_none());
        assert!(value.get("tokensUsed").is_none());
        assert!(value.get("expandable").is_none());
    }

    #[test]
    fn test_schema_has_token_budget() {
        let schema_value = schema();
        let tb = &schema_value["properties"]["token_budget"];
        assert!(tb.is_object());
        assert_eq!(tb["minimum"], 100);
        assert_eq!(tb["maximum"], 100000);
    }

    // ========================================================================
    // TAG_PREFIX TESTS (PR1)
    // ========================================================================

    #[test]
    fn test_tags_match_prefix_unit() {
        let with_meeting = vec!["meeting:standup".to_string(), "team".to_string()];
        let without_meeting = vec!["adhoc".to_string(), "team".to_string()];
        let tagless: Vec<String> = vec![];

        assert!(tags_match_prefix(&with_meeting, "meeting:"));
        assert!(!tags_match_prefix(&without_meeting, "meeting:"));
        // Empty prefix matches when any tag exists; never matches a tagless
        // memory. This preserves the "tag_prefix is a filter, not a default
        // wildcard" semantics — a tagless memory has no tag-prefix to satisfy.
        assert!(tags_match_prefix(&with_meeting, ""));
        assert!(!tags_match_prefix(&tagless, ""));
        // Case-sensitive (consistent with existing exact-tag matching).
        assert!(!tags_match_prefix(&with_meeting, "Meeting:"));
        // Prefix must match from the start, not anywhere in the tag value.
        assert!(!tags_match_prefix(&with_meeting, "standup"));
    }

    #[test]
    fn test_schema_has_tag_prefix() {
        let schema_value = schema();
        let tp = &schema_value["properties"]["tag_prefix"];
        assert!(tp.is_object(), "tag_prefix property must be present");
        assert_eq!(tp["type"], "string");
        // tag_prefix is NOT required.
        let required = schema_value["required"].as_array().unwrap();
        assert!(!required.contains(&serde_json::json!("tag_prefix")));
    }

    // ===================== #57 Phase 4 source filters =====================

    /// Build a KnowledgeNode carrying a source envelope for filter tests.
    fn node_with_source(
        system: &str,
        project: &str,
        id: &str,
        author: &str,
        updated: &str,
    ) -> vestige_core::KnowledgeNode {
        let mut n = vestige_core::KnowledgeNode::default();
        n.id = format!("{system}-{id}");
        // SourceEnvelope is #[non_exhaustive]; build via Default + field set.
        let mut env = vestige_core::SourceEnvelope::default();
        env.source_system = Some(system.to_string());
        env.source_id = Some(id.to_string());
        env.source_url = Some(format!("https://x/{id}"));
        env.source_updated_at = chrono::DateTime::parse_from_rfc3339(updated)
            .ok()
            .map(|d| d.with_timezone(&chrono::Utc));
        env.content_hash = Some("h".to_string());
        env.source_project = Some(project.to_string());
        env.source_type = Some("issue".to_string());
        env.source_author = Some(author.to_string());
        n.source_envelope = Some(env);
        n
    }

    fn filter_from(json: serde_json::Value) -> SourceFilter {
        let mut v = json;
        v["query"] = serde_json::json!("q");
        let args: SearchArgs = serde_json::from_value(v).unwrap();
        SourceFilter::from_args(&args).unwrap()
    }

    #[test]
    fn source_filter_empty_matches_everything() {
        let f = SourceFilter::default();
        assert!(!f.is_active());
        let gh = node_with_source("github", "o/r", "1", "octo", "2026-06-19T00:00:00Z");
        let legacy = vestige_core::KnowledgeNode::default(); // no envelope
        assert!(node_matches_source(&gh, &f));
        assert!(node_matches_source(&legacy, &f), "no filter = unchanged");
    }

    #[test]
    fn source_filter_exact_fields() {
        let gh = node_with_source("github", "o/r", "57", "octo", "2026-06-19T00:00:00Z");
        let rm = node_with_source("redmine", "infra", "57", "jane", "2026-06-19T00:00:00Z");

        let by_system = filter_from(serde_json::json!({"sourceSystem": "github"}));
        assert!(node_matches_source(&gh, &by_system));
        assert!(!node_matches_source(&rm, &by_system));

        let by_project = filter_from(serde_json::json!({"sourceProject": "infra"}));
        assert!(node_matches_source(&rm, &by_project));
        assert!(!node_matches_source(&gh, &by_project));

        let by_author = filter_from(serde_json::json!({"sourceAuthor": "octo"}));
        assert!(node_matches_source(&gh, &by_author));
        assert!(!node_matches_source(&rm, &by_author));

        // id + system together disambiguate across systems sharing an id.
        let by_id_sys =
            filter_from(serde_json::json!({"sourceSystem": "redmine", "sourceId": "57"}));
        assert!(node_matches_source(&rm, &by_id_sys));
        assert!(!node_matches_source(&gh, &by_id_sys));
    }

    #[test]
    fn source_filter_excludes_legacy_memories_when_envelope_scoped() {
        let legacy = vestige_core::KnowledgeNode::default();
        let f = filter_from(serde_json::json!({"sourceSystem": "github"}));
        assert!(
            !node_matches_source(&legacy, &f),
            "an envelope-scoped filter must exclude memories with no source"
        );
    }

    #[test]
    fn source_filter_date_bounds_inclusive() {
        let n = node_with_source("github", "o/r", "1", "octo", "2026-06-15T12:00:00Z");
        // After bound: inclusive at the exact instant, excludes earlier.
        assert!(node_matches_source(
            &n,
            &filter_from(serde_json::json!({"sourceUpdatedAfter": "2026-06-15T12:00:00Z"}))
        ));
        assert!(!node_matches_source(
            &n,
            &filter_from(serde_json::json!({"sourceUpdatedAfter": "2026-06-16T00:00:00Z"}))
        ));
        // Before bound: inclusive, excludes later.
        assert!(node_matches_source(
            &n,
            &filter_from(serde_json::json!({"sourceUpdatedBefore": "2026-06-15T12:00:00Z"}))
        ));
        assert!(!node_matches_source(
            &n,
            &filter_from(serde_json::json!({"sourceUpdatedBefore": "2026-06-15T00:00:00Z"}))
        ));
    }

    #[test]
    fn source_filter_status_valid_vs_tombstoned() {
        let mut live = node_with_source("github", "o/r", "1", "octo", "2026-06-19T00:00:00Z");
        let mut dead = node_with_source("github", "o/r", "2", "octo", "2026-06-19T00:00:00Z");
        let legacy = vestige_core::KnowledgeNode::default();
        // Tombstone `dead` by setting valid_until in the past.
        dead.valid_until = Some(chrono::Utc::now() - chrono::Duration::days(1));
        live.valid_until = None;

        let valid = filter_from(serde_json::json!({"sourceStatus": "valid"}));
        assert!(node_matches_source(&live, &valid));
        assert!(!node_matches_source(&dead, &valid));
        assert!(
            !node_matches_source(&legacy, &valid),
            "source_status is source-scoped and must not include legacy memories"
        );

        let tomb = filter_from(serde_json::json!({"sourceStatus": "tombstoned"}));
        assert!(!node_matches_source(&live, &tomb));
        assert!(node_matches_source(&dead, &tomb));
        assert!(!node_matches_source(&legacy, &tomb));
    }

    #[test]
    fn source_filter_rejects_bad_timestamp_and_status() {
        let mut v = serde_json::json!({"query": "q", "sourceUpdatedAfter": "not-a-date"});
        let args: SearchArgs = serde_json::from_value(v.take()).unwrap();
        assert!(SourceFilter::from_args(&args).is_err());

        let mut v2 = serde_json::json!({"query": "q", "sourceStatus": "bogus"});
        let args2: SearchArgs = serde_json::from_value(v2.take()).unwrap();
        assert!(SourceFilter::from_args(&args2).is_err());
    }

    #[test]
    fn test_schema_has_source_filters() {
        let s = schema();
        for prop in [
            "source_system",
            "source_project",
            "source_id",
            "source_type",
            "source_author",
            "source_updated_after",
            "source_updated_before",
            "source_status",
        ] {
            assert!(
                s["properties"][prop].is_object(),
                "schema must expose {prop}"
            );
        }
        // None of the source filters are required.
        let required = s["required"].as_array().unwrap();
        for prop in ["source_system", "source_status"] {
            assert!(!required.contains(&serde_json::json!(prop)));
        }
    }

    /// Helper that ingests a memory with specific tags. The base
    /// `ingest_test_content` helper passes `tags: vec![]`, which is fine
    /// for legacy tests but not for tag_prefix coverage.
    async fn ingest_with_tags(storage: &Arc<Storage>, content: &str, tags: Vec<&str>) -> String {
        let input = IngestInput {
            content: content.to_string(),
            node_type: "fact".to_string(),
            source: None,
            sentiment_score: 0.0,
            sentiment_magnitude: 0.0,
            tags: tags.into_iter().map(String::from).collect(),
            valid_from: None,
            valid_until: None,
            source_envelope: None,
        };
        let node = storage.ingest(input).unwrap();
        node.id
    }

    #[tokio::test]
    async fn test_search_tag_prefix_filters_results() {
        let (storage, _dir) = test_storage().await;
        // Three memories matching the query semantically, only two carry
        // the meeting:* tag-class.
        ingest_with_tags(
            &storage,
            "Standup discussion about Q3 roadmap blockers",
            vec!["meeting:standup", "roadmap"],
        )
        .await;
        ingest_with_tags(
            &storage,
            "1-on-1 sync on roadmap clarity and ownership",
            vec!["meeting:1-on-1", "roadmap"],
        )
        .await;
        ingest_with_tags(
            &storage,
            "Solo note: roadmap dependency graph audit",
            vec!["adhoc", "roadmap"],
        )
        .await;

        let args = serde_json::json!({
            "query": "roadmap",
            "tag_prefix": "meeting:",
            "min_similarity": 0.0
        });
        let result = execute(
            &storage,
            &test_cognitive(),
            &OutputConfig::default(),
            Some(args),
        )
        .await;
        assert!(result.is_ok(), "{:?}", result);
        let value = result.unwrap();
        let results = value["results"].as_array().unwrap();
        // Both meeting:* memories should land; the adhoc one should not.
        for r in results {
            let tags = r["tags"].as_array().expect("tags must be present");
            let has_meeting = tags
                .iter()
                .any(|t| t.as_str().is_some_and(|s| s.starts_with("meeting:")));
            assert!(has_meeting, "result lacks meeting:* tag: {}", r);
        }
        // We expect 2 matches given the corpus above. The exact count
        // depends on the cognitive pipeline's competition/suppression
        // dynamics, so assert a lower bound.
        assert!(
            !results.is_empty(),
            "tag_prefix should leave at least one meeting:* result, got {}",
            results.len()
        );
    }

    #[tokio::test]
    async fn test_search_tag_prefix_excludes_tagless_memories() {
        let (storage, _dir) = test_storage().await;
        ingest_with_tags(
            &storage,
            "Notebook entry about consolidation cycles",
            vec![], // tagless
        )
        .await;
        ingest_with_tags(
            &storage,
            "Project note about consolidation cycles",
            vec!["project:vestige"],
        )
        .await;

        let args = serde_json::json!({
            "query": "consolidation",
            "tag_prefix": "project:",
            "min_similarity": 0.0
        });
        let result = execute(
            &storage,
            &test_cognitive(),
            &OutputConfig::default(),
            Some(args),
        )
        .await;
        assert!(result.is_ok());
        let value = result.unwrap();
        let results = value["results"].as_array().unwrap();
        for r in results {
            let tags = r["tags"].as_array().expect("tags must be present");
            let has_project = tags
                .iter()
                .any(|t| t.as_str().is_some_and(|s| s.starts_with("project:")));
            assert!(has_project, "tagless or non-project result leaked: {}", r);
        }
    }

    #[tokio::test]
    async fn test_search_without_tag_prefix_unchanged() {
        // Backwards-compat: same corpus, same query, no tag_prefix → all
        // results pass through regardless of tag composition. This is the
        // load-bearing test for additive-only behavior.
        let (storage, _dir) = test_storage().await;
        ingest_with_tags(&storage, "Notebook entry about audit cycles", vec![]).await;
        ingest_with_tags(
            &storage,
            "Project note about audit cycles",
            vec!["project:audit"],
        )
        .await;

        let args = serde_json::json!({
            "query": "audit",
            "min_similarity": 0.0
        });
        let result = execute(
            &storage,
            &test_cognitive(),
            &OutputConfig::default(),
            Some(args),
        )
        .await;
        assert!(result.is_ok());
        let value = result.unwrap();
        let results = value["results"].as_array().unwrap();
        // Both should be retrievable since no tag_prefix is set.
        assert!(
            !results.is_empty(),
            "expected at least one result with no tag_prefix"
        );
    }

    #[tokio::test]
    async fn test_search_tag_prefix_concrete_path() {
        // Concrete-search path (literal query) must also honor tag_prefix.
        let (storage, _dir) = test_storage().await;
        ingest_with_tags(
            &storage,
            "OPENAI_API_KEY rotation playbook for meetings",
            vec!["meeting:ops"],
        )
        .await;
        ingest_with_tags(
            &storage,
            "OPENAI_API_KEY rotation playbook for solo audits",
            vec!["adhoc"],
        )
        .await;

        let args = serde_json::json!({
            "query": "OPENAI_API_KEY",
            "concrete": true,
            "tag_prefix": "meeting:"
        });
        let result = execute(
            &storage,
            &test_cognitive(),
            &OutputConfig::default(),
            Some(args),
        )
        .await;
        assert!(result.is_ok(), "{:?}", result);
        let value = result.unwrap();
        assert_eq!(value["method"], "concrete");
        let results = value["results"].as_array().unwrap();
        for r in results {
            let tags = r["tags"].as_array().expect("tags must be present");
            let has_meeting = tags
                .iter()
                .any(|t| t.as_str().is_some_and(|s| s.starts_with("meeting:")));
            assert!(has_meeting, "concrete result lacks meeting:* tag: {}", r);
        }
    }

    // ========================================================================
    // Phase 2: Configurable Output — precedence tests
    // ========================================================================

    /// Config-file detail_level applies when no explicit MCP param is given.
    #[tokio::test]
    async fn test_config_detail_level_applies_without_param() {
        let (storage, _dir) = test_storage().await;
        ingest_test_content(&storage, "Config detail level fallback content.").await;

        // Config selects `full`; the call passes no detail_level.
        let cfg = vestige_core::VestigeConfig::parse("[defaults]\ndetail_level=\"full\"").output();
        let args = serde_json::json!({ "query": "config detail", "min_similarity": 0.0 });
        let value = execute(&storage, &test_cognitive(), &cfg, Some(args))
            .await
            .unwrap();
        assert_eq!(value["detailLevel"], "full");
    }

    /// Explicit MCP param beats the config file (precedence layer 1 > 2).
    #[tokio::test]
    async fn test_explicit_param_overrides_config() {
        let (storage, _dir) = test_storage().await;
        ingest_test_content(&storage, "Explicit overrides config content.").await;

        // Config says `full`, but the call explicitly requests `brief`.
        let cfg = vestige_core::VestigeConfig::parse("[defaults]\ndetail_level=\"full\"").output();
        let args = serde_json::json!({
            "query": "explicit override",
            "detail_level": "brief",
            "min_similarity": 0.0
        });
        let value = execute(&storage, &test_cognitive(), &cfg, Some(args))
            .await
            .unwrap();
        assert_eq!(value["detailLevel"], "brief");
    }

    /// The `lean` profile masks scores and timestamps from results.
    #[tokio::test]
    async fn test_lean_profile_masks_scores_and_timestamps() {
        let (storage, _dir) = test_storage().await;
        ingest_test_content(&storage, "Lean profile masking content.").await;

        let cfg = vestige_core::VestigeConfig::parse("[defaults]\nprofile=lean").output();
        let args = serde_json::json!({ "query": "lean masking", "min_similarity": 0.0 });
        let value = execute(&storage, &test_cognitive(), &cfg, Some(args))
            .await
            .unwrap();
        assert_eq!(value["profile"], "lean");
        if let Some(first) = value["results"].as_array().and_then(|a| a.first()) {
            assert!(
                first.get("combinedScore").is_none(),
                "lean must drop scores"
            );
            assert!(
                first.get("createdAt").is_none(),
                "lean must drop timestamps"
            );
        }
    }

    /// The default profile is byte-for-byte the historical behavior: summary
    /// detail with scores and timestamps present.
    #[tokio::test]
    async fn test_default_profile_preserves_behavior() {
        let (storage, _dir) = test_storage().await;
        ingest_test_content(&storage, "Default profile preserved content.").await;

        let args = serde_json::json!({ "query": "default preserved", "min_similarity": 0.0 });
        let value = execute(
            &storage,
            &test_cognitive(),
            &OutputConfig::default(),
            Some(args),
        )
        .await
        .unwrap();
        assert_eq!(value["detailLevel"], "summary");
        assert_eq!(value["profile"], "default");
        if let Some(first) = value["results"].as_array().and_then(|a| a.first()) {
            assert!(first.get("createdAt").is_some(), "default keeps timestamps");
        }
    }
}
