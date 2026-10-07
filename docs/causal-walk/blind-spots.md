# Causal walk blind spots

239 gaps between a failure report and the commit `causal_walk` can defend.
Each row is one catalog item. The function is where a fix lands on this
branch, read from the current tree. The catalog's 4.2.0 line numbers were
not copied.

A link is a recorded causal event: a commit touching a path, a blame, a
revert, a tool call, a derivation, a supersession, a gate decision, or a CI
result. Paths and ids join only by exact bytes.

## Status counts

| Status | Count |
| --- | ---: |
| implemented in this PR | 5 |
| buildable now | 34 |
| needs a new receipt/event type | 195 |
| no allowed fix | 5 |
| **total** | **239** |

Catalog priority, unchanged: 139 high, 84 medium, 16 low.

## How a status was assigned

- `implemented in this PR` — items 10, 12, 20, 21, and 22. Each has a passing test.
- `buildable now` — the log already stores the edge, or local git already prints the fact (`merge-base`, blame, a diff header, a trailer, a patch-id) and nothing new has to be fetched. High items in this set that this PR does not implement have an `#[ignore]`d test named with the catalog number.
- `needs a new receipt/event type` — the discriminating fact is not in the log and is not a local git object the walk already reads. The name in parentheses is that receipt. Until it is stored, the walk has nothing exact to follow.
- `no allowed fix` — items 19, 60, 66, 70, and 114. Closing them would join on a symbol, scan a sentence for a SHA, normalize path bytes, or blame a resource address onto a file.

Item 20 also refuses the write in `refuse_foreign_codebase` (`crates/vestige-mcp/src/tools/smart_ingest.rs`) and follows the new `touched` edge from `resolve_git_frames` (`crates/vestige-mcp/src/tools/causal_walk.rs`). Item 10's whitespace classification is `line_change_is_blank_or_comment` in `crates/vestige-mcp/src/tools/repo_ingest.rs`. Item 22's old-side edge is written by `record_git_edges` in the same file. Item 21's trailer parse is `fixes_targets` in `crates/vestige-core/src/advanced/git_records.rs`; expansion is only `git rev-parse --verify <token>^{commit}`.

## Package paths, lockfiles, and upstream

- **1. Published package path vs repo source path** (high) — `crates/vestige-mcp/src/tools/causal_walk.rs` `local_crate_sha` — needs a new receipt/event type (`published_path`)
- **2. Monorepo package boundaries** (high) — `crates/vestige-mcp/src/tools/repo_ingest.rs` `check_top_of_work_tree` — needs a new receipt/event type (`workspace_member`)
- **4. Dependency bumps that change behavior without touching the app file** (high) — `crates/vestige-core/src/advanced/git_records.rs` `lock_bumps_from_diff` — buildable now
- **16. Vendored copies** (medium) — `crates/vestige-mcp/src/tools/repo_ingest.rs` `upstream_range` — buildable now
- **41. Feature unification from another workspace member** (high) — `crates/vestige-mcp/src/tools/causal_walk.rs` `packages_on_parent_chain` — needs a new receipt/event type (`workspace_member`)
- **54. Lock-bump cap drops a package** (low) — `crates/vestige-core/src/advanced/git_records.rs` `parse_git_log` — buildable now
- **97. Advisory text with no path** (high) — `crates/vestige-mcp/src/auto_connect.rs` `classify_marked_token` — needs a new receipt/event type (`advisory_anchor`)
- **107. Receipt lock hash is not the git blob** (high) — `crates/vestige-core/src/advanced/git_records.rs` `lock_bumps_from_diff` — needs a new receipt/event type (`lock_hash`)
- **109. Manifest edited, lock opened, lock blob unchanged** (high) — `crates/vestige-core/src/advanced/git_records.rs` `lock_bumps_from_diff` — needs a new receipt/event type (`opened_path`)

## Imports and cross-file breaks

- **3. Test-only edits masking a production regression** (high) — `crates/vestige-core/src/advanced/git_records.rs` `import_target` — needs a new receipt/event type (`import_edge`)
- **9. Cross-file break: the frame's last toucher is not the breaking commit** (high) — `crates/vestige-mcp/src/tools/causal_walk.rs` `walk_from` — needs a new receipt/event type (`import_edge`)
- **24. Inducing commit shares no file with the crash** (high) — `crates/vestige-core/src/advanced/git_records.rs` `import_target` — needs a new receipt/event type (`import_edge`)
- **40. Quoted import paths the parser does not read** (high) — `crates/vestige-core/src/advanced/git_records.rs` `import_target` — needs a new receipt/event type (`import_edge`)
- **92. Null from a file the crash does not include** (medium) — `crates/vestige-mcp/src/tools/repo_ingest.rs` `record_git_edges` — needs a new receipt/event type (`import_edge`)

## Blame, hunks, and rank

- **10. Semantic change with no line-level blame** (high) — `crates/vestige-mcp/src/tools/causal_walk.rs` `line_slots` — implemented in this PR
- **22. Omission fault (remove-mapping ghost)** (high) — `crates/vestige-core/src/advanced/git_records.rs` `parse_hunk_ranges` — implemented in this PR
- **23. Deletion-only inducing commit (add-mapping ghost)** (high) — `crates/vestige-mcp/src/tools/causal_walk.rs` `git_rank` — needs a new receipt/event type (`prove_verdict`)
- **25. Refactoring commit owns the blamed line** (high) — `crates/vestige-mcp/src/tools/repo_ingest.rs` `blame_at` — needs a new receipt/event type (`line_history`)
- **26. Tangled commit** (medium) — `crates/vestige-mcp/src/walk_verify/hunks.rs` `ddmin` — needs a new receipt/event type (`prove_kept_hunks`)
- **37. A 1-minimal hunk set is not the causal line** (medium) — `crates/vestige-mcp/src/walk_verify/hunks.rs` `ddmin` — buildable now
- **50. A whitespace-only diff is the break** (medium) — `crates/vestige-mcp/src/tools/repo_ingest.rs` `line_change_is_blank_or_comment` — needs a new receipt/event type (`prove_verdict`)
- **53. One line number is blamed on every matched path** (high) — `crates/vestige-mcp/src/tools/causal_walk.rs` `resolve_git_frames` — buildable now
- **62. Two edits on one source line** (low) — `crates/vestige-mcp/src/tools/repo_ingest.rs` `blame_at` — buildable now
- **91. Off-by-one on the neighboring line** (medium) — `crates/vestige-mcp/src/tools/causal_walk.rs` `touched_line` — buildable now

## Version range, merges, and ancestry

- **11. Merge commits, including conflict-only resolutions** (low) — `crates/vestige-mcp/src/tools/repo_ingest.rs` `read_commits` — buildable now
- **12. `version_range` drops the commit that arrived on a second parent** (high) — `crates/vestige-mcp/src/tools/causal_walk.rs` `commit_in_ancestor_range` — implemented in this PR
- **13. Squash merges** (high) — `crates/vestige-mcp/src/tools/repo_ingest.rs` `record_patch_identities` — buildable now
- **32. Revert of a revert** (high) — `crates/vestige-mcp/src/tools/causal_walk.rs` `reverted_after` — buildable now
- **33. Bisect skip next to the first bad commit** (high) — `crates/vestige-mcp/src/walk_verify/git.rs` `first_bad_logged` — buildable now
- **34. The merge-base is already bad** (high) — `crates/vestige-mcp/src/walk_verify/run.rs` `bisect_start` — needs a new receipt/event type (`prove_verdict`)
- **35. Old revisions do not build without a hotfix** (medium) — `crates/vestige-mcp/src/walk_verify/run.rs` `bisect_start` — needs a new receipt/event type (`hotfix_sha`)
- **36. Several recorded good revisions** (medium) — `crates/vestige-mcp/src/tools/causal_walk.rs` `apply_version_range` — needs a new receipt/event type (`ci_result`)
- **117. Untested SHA gap** (high) — `crates/vestige-mcp/src/walk_verify/git.rs` `first_bad_logged` — buildable now

## Renames, modes, blobs, and paths

- **14. Renames** (medium) — `crates/vestige-core/src/advanced/git_records.rs` `diff_payload` — buildable now
- **15. Generated files** (medium) — `crates/vestige-core/src/advanced/git_records.rs` `parse_git_log` — needs a new receipt/event type (`generated_from`)
- **43. Submodule gitlink** (high) — `crates/vestige-core/src/advanced/git_records.rs` `parse_git_log` — buildable now
- **44. LFS pointer with textconv forced off** (high) — `crates/vestige-core/src/advanced/git_records.rs` `diff_payload` — buildable now
- **45. Binary diff records no hunk and no blob id** (medium) — `crates/vestige-core/src/advanced/git_records.rs` `diff_payload` — needs a new receipt/event type (`blob_oid`)
- **46. Mode-only change** (medium) — `crates/vestige-core/src/advanced/git_records.rs` `diff_payload` — buildable now
- **47. Symlink retarget** (medium) — `crates/vestige-core/src/advanced/git_records.rs` `diff_payload` — buildable now
- **48. Same blob removed and added under another path** (medium) — `crates/vestige-core/src/advanced/git_records.rs` `parse_git_log` — buildable now
- **49. Case differs and git recorded no rename** (medium) — `crates/vestige-mcp/src/tools/causal_walk.rs` `paths_identify` — buildable now
- **51. The checkout has no readable history** (high) — `crates/vestige-mcp/src/tools/repo_ingest.rs` `check_top_of_work_tree` — buildable now
- **52. Git log output is cut off** (medium) — `crates/vestige-mcp/src/tools/repo_ingest.rs` `run_git` — buildable now
- **67. Replace refs hide the SHA the failure names** (medium) — `crates/vestige-mcp/src/tools/repo_ingest.rs` `read_commits` — buildable now
- **70. Path bytes differ by Unicode normalization** (medium) — `crates/vestige-mcp/src/tools/causal_walk.rs` `normalize_path` — no allowed fix
- **102. Path outside the work tree** (high) — `crates/vestige-mcp/src/tools/causal_walk.rs` `paths_identify` — buildable now

## Failure attachment and trailers

- **19. Symbol-only and bare-test reports** (high) — `crates/vestige-mcp/src/tools/causal_walk.rs` `failing_test_path` — no allowed fix
- **20. The failure record is not tied to the file anchor or the version window** (high) — `crates/vestige-mcp/src/auto_connect.rs` `classify_marked_token` — implemented in this PR
- **21. Abbreviated `Fixes:` trailer** (high) — `crates/vestige-mcp/src/tools/repo_ingest.rs` `resolve_fix_shas` — implemented in this PR
- **38. The first stack frame is the wrong file** (high) — `crates/vestige-mcp/src/tools/causal_walk.rs` `split_frame` — buildable now
- **39. Locator syntax other than `path:line`** (high) — `crates/vestige-mcp/src/tools/causal_walk.rs` `split_frame` — buildable now
- **63. Issue trailer without an `owner/repo` token** (high) — `crates/vestige-core/src/advanced/git_records.rs` `fixes_targets` — buildable now
- **66. A full SHA written in a sentence** (low) — `crates/vestige-core/src/advanced/git_records.rs` `fixes_targets` — no allowed fix

## Time, cherry-pick, and patch identity

- **28. Author time after the failure drops the cause** (high) — `crates/vestige-mcp/src/tools/causal_walk.rs` `git_admissible` — buildable now
- **29. Author date orders a rebased or cherry-picked commit** (medium) — `crates/vestige-mcp/src/tools/causal_walk.rs` `git_rank` — buildable now
- **30. Conflicted cherry-pick** (high) — `crates/vestige-mcp/src/tools/repo_ingest.rs` `record_patch_identities` — buildable now
- **31. Partial revert writes a whole-commit `corrects`** (medium) — `crates/vestige-mcp/src/tools/repo_ingest.rs` `record_git_edges` — buildable now
- **69. Patch-id skipped for a merge** (medium) — `crates/vestige-mcp/src/tools/repo_ingest.rs` `record_patch_identities` — buildable now

## Frames without a repo path

- **8. Minified or bundled frames** (high) — `crates/vestige-mcp/src/tools/causal_walk.rs` `paths_identify` — needs a new receipt/event type (`source_map`)
- **60. Macro or generated line versus the definition** (medium) — `crates/vestige-mcp/src/tools/repo_ingest.rs` `blame_at` — no allowed fix
- **95. Profile names a function and no path** (medium) — `crates/vestige-mcp/src/tools/causal_walk.rs` `split_frame` — needs a new receipt/event type (`profile_frame`)
- **111. UI receipt has no repo path** (high) — `crates/vestige-mcp/src/tools/causal_walk.rs` `split_frame` — needs a new receipt/event type (`ui_path`)
- **171. Sanitizer frame is an address and a module offset** (high) — `crates/vestige-mcp/src/tools/repo_ingest.rs` `blame_at` — needs a new receipt/event type (`symbolizer_frame`)
- **172. Symbolizer receipt has no line** (high) — `crates/vestige-mcp/src/tools/causal_walk.rs` `git_rank` — needs a new receipt/event type (`symbolizer_frame`)
- **211. Fast unwinder stops at a frame with no frame pointer** (high) — `crates/vestige-mcp/src/tools/repo_ingest.rs` `blame_at` — needs a new receipt/event type (`stack_receipt`)

## CI runs, caches, and checkouts

- **7. Flaky tests** (high) — `crates/vestige-mcp/src/tools/causal_walk.rs` `classify_starts` — needs a new receipt/event type (`ci_result`)
- **18. Runtime, toolchain, and image changes** (medium) — `crates/vestige-mcp/src/tools/causal_walk.rs` `classify_starts` — needs a new receipt/event type (`ci_result`)
- **42. Same SHA, two recorded commands, two verdicts** (high) — `crates/vestige-mcp/src/tools/causal_walk.rs` `git_rank` — needs a new receipt/event type (`tool_argv`)
- **57. Live object drift, no new commit** (high) — `crates/vestige-mcp/src/tools/causal_walk.rs` `classify_starts` — needs a new receipt/event type (`tool_result`)
- **58. Test-order dependence** (high) — `crates/vestige-mcp/src/walk_verify/run.rs` `probe` — needs a new receipt/event type (`tool_argv`)
- **59. A hang is counted as cannot-test** (high) — `crates/vestige-mcp/src/walk_verify/probe.rs` `run_test` — buildable now
- **64. A green CI run did not execute the failing path** (high) — `crates/vestige-mcp/src/tools/causal_walk.rs` `classify_starts` — needs a new receipt/event type (`workflow_paths`)
- **65. Unpinned image tag, Dockerfile unchanged** (high) — `crates/vestige-mcp/src/tools/causal_walk.rs` `upstream_note_for` — needs a new receipt/event type (`image_digest`)
- **71. Synthetic pull-request merge SHA** (high) — `crates/vestige-mcp/src/tools/repo_ingest.rs` `read_commits` — needs a new receipt/event type (`ci_event_sha`)
- **72. Merge-group SHA** (high) — `crates/vestige-mcp/src/tools/causal_walk.rs` `apply_version_range` — needs a new receipt/event type (`ci_event_sha`)
- **73. Required check absent or skipped** (high) — `crates/vestige-mcp/src/tools/causal_walk.rs` `classify_starts` — needs a new receipt/event type (`check_conclusion`)
- **74. Parent success over a failed or cancelled child** (high) — `crates/vestige-mcp/src/tools/causal_walk.rs` `git_rank` — needs a new receipt/event type (`job_conclusion`)
- **75. Dependent job never ran** (high) — `crates/vestige-mcp/src/tools/causal_walk.rs` `classify_starts` — needs a new receipt/event type (`job_dependency`)
- **76. Retry omits dependency job ids** (high) — `crates/vestige-mcp/src/tools/causal_walk.rs` `classify_starts` — needs a new receipt/event type (`attempt_job_ids`)
- **77. Cache key bytes differ** (high) — `crates/vestige-mcp/src/tools/causal_walk.rs` `classify_starts` — needs a new receipt/event type (`cache_key`)
- **78. Hit boolean true while the keys differ** (medium) — `crates/vestige-mcp/src/tools/causal_walk.rs` `classify_starts` — needs a new receipt/event type (`cache_key`)
- **79. Cache miss recorded, step still success** (high) — `crates/vestige-mcp/src/walk_verify/run.rs` `prove` — needs a new receipt/event type (`cache_hit`)
- **80. Artifact from another run id** (medium) — `crates/vestige-mcp/src/tools/causal_walk.rs` `classify_starts` — needs a new receipt/event type (`artifact_run`)
- **81. Step failed, job succeeded** (high) — `crates/vestige-mcp/src/tools/causal_walk.rs` `classify_starts` — needs a new receipt/event type (`step_conclusion`)
- **82. Event-field bytes differ** (high) — `crates/vestige-mcp/src/tools/causal_walk.rs` `classify_starts` — needs a new receipt/event type (`event_name`)
- **83. Checkout HEAD is not the event SHA** (high) — `crates/vestige-mcp/src/tools/causal_walk.rs` `failure_revision` — needs a new receipt/event type (`checkout_sha`)
- **84. Selection file omitted the failing path** (high) — `crates/vestige-mcp/src/tools/causal_walk.rs` `classify_starts` — needs a new receipt/event type (`selection_paths`)
- **85. Failing check recorded no command** (medium) — `crates/vestige-mcp/src/walk_verify/run.rs` `prove` — needs a new receipt/event type (`check_argv`)
- **86. Cancelled siblings are not passes** (high) — `crates/vestige-mcp/src/tools/causal_walk.rs` `classify_starts` — needs a new receipt/event type (`job_conclusion`)
- **87. pull_request_target ran the default-branch SHA** (high) — `crates/vestige-mcp/src/tools/causal_walk.rs` `classify_starts` — needs a new receipt/event type (`ci_event_sha`)
- **88. Sparse checkout path list** (medium) — `crates/vestige-mcp/src/tools/causal_walk.rs` `classify_starts` — needs a new receipt/event type (`checkout_paths`)
- **89. Both parents passed, the merge failed** (high) — `crates/vestige-mcp/src/tools/causal_walk.rs` `git_rank` — needs a new receipt/event type (`parent_verdict`)
- **90. Tested tree is not the squash tree** (high) — `crates/vestige-mcp/src/tools/causal_walk.rs` `classify_starts` — needs a new receipt/event type (`tree_oid`)
- **94. Overlapping writers of one path** (high) — `crates/vestige-mcp/src/tools/causal_walk.rs` `git_rank` — needs a new receipt/event type (`overlapping_receipt`)

## Frames without a repo path


## Receipts the walk does not read

- **5. Config-only commits** (medium) — `crates/vestige-mcp/src/tools/repo_ingest.rs` `record_git_edges` — needs a new receipt/event type (`config_input_path`)
- **6. Build-tool changes** (medium) — `crates/vestige-mcp/src/tools/repo_ingest.rs` `log_args` — needs a new receipt/event type (`build_input_path`)
- **17. Multi-commit regressions** (medium) — `crates/vestige-mcp/src/walk_verify/hunks.rs` `ddmin` — needs a new receipt/event type (`prove_pair`)
- **27. Coincidental correctness** (medium) — `crates/vestige-mcp/src/tools/causal_walk.rs` `classify_starts` — needs a new receipt/event type (`coverage_hit`)
- **55. `Fixes:` trailer cap drops a SHA** (low) — `crates/vestige-core/src/advanced/git_records.rs` `parse_git_log` — buildable now
- **56. Walk depth cuts a lineage chain** (medium) — `crates/vestige-mcp/src/tools/causal_walk.rs` `walk_from` — buildable now
- **61. Build-constraint line** (medium) — `crates/vestige-mcp/src/tools/causal_walk.rs` `git_rank` — needs a new receipt/event type (`tool_argv`)
- **68. Empty commit** (low) — `crates/vestige-mcp/src/tools/repo_ingest.rs` `record_git_edges` — needs a new receipt/event type (`prove_verdict`)
- **93. Coercion with no recorded test** (low) — `crates/vestige-mcp/src/tools/causal_walk.rs` `git_rank` — needs a new receipt/event type (`prove_verdict`)
- **96. Benchmark receipt has no path** (low) — `crates/vestige-mcp/src/tools/causal_walk.rs` `git_rank` — needs a new receipt/event type (`benchmark_path`)
- **98. Numeric result with no recorded flip** (low) — `crates/vestige-mcp/src/tools/repo_ingest.rs` `blame_at` — needs a new receipt/event type (`prove_verdict`)
- **99. Schedule instant moved** (medium) — `crates/vestige-mcp/src/tools/causal_walk.rs` `git_admissible` — needs a new receipt/event type (`schedule_instant`)
- **100. Locale bytes differ and no path is named** (low) — `crates/vestige-mcp/src/tools/causal_walk.rs` `git_rank` — needs a new receipt/event type (`env_bytes`)
- **101. Decoded line number on a non-UTF-8 blob** (medium) — `crates/vestige-mcp/src/tools/repo_ingest.rs` `blame_at` — needs a new receipt/event type (`byte_offset`)
- **103. Same URL, status changed, SHA unchanged** (high) — `crates/vestige-mcp/src/walk_verify/run.rs` `prove` — needs a new receipt/event type (`http_status_receipt`)
- **104. Body hash changed, schema path absent** (medium) — `crates/vestige-mcp/src/auto_connect.rs` `classify_marked_token` — needs a new receipt/event type (`body_hash`)
- **105. Migration id with no file** (medium) — `crates/vestige-mcp/src/tools/causal_walk.rs` `classify_starts` — needs a new receipt/event type (`migration_id`)
- **106. Flag map differs** (medium) — `crates/vestige-mcp/src/tools/causal_walk.rs` `git_rank` — needs a new receipt/event type (`flag_map`)
- **108. Tool version string differs** (low) — `crates/vestige-mcp/src/tools/causal_walk.rs` `git_rank` — needs a new receipt/event type (`tool_version`)
- **110. OS id differs** (medium) — `crates/vestige-mcp/src/tools/causal_walk.rs` `classify_starts` — needs a new receipt/event type (`os_id`)
- **112. State serial moved, config blob did not** (high) — `crates/vestige-mcp/src/walk_verify/run.rs` `prove` — needs a new receipt/event type (`state_serial`)
- **113. Exit 2 disagrees with the plan JSON** (high) — `crates/vestige-mcp/src/walk_verify/probe.rs` `verdict_of` — needs a new receipt/event type (`plan_json`)
- **114. Provider disagreed with its plan** (medium) — `crates/vestige-mcp/src/walk_verify/run.rs` `prove` — no allowed fix
- **115. An earlier tool call wrote the failing path** (high) — `crates/vestige-mcp/src/tools/causal_walk.rs` `classify_starts` — needs a new receipt/event type (`tool_write`)
- **116. Model id differs** (high) — `crates/vestige-mcp/src/tools/causal_walk.rs` `git_rank` — needs a new receipt/event type (`model_id`)
- **118. Dispatch inputs differ** (medium) — `crates/vestige-mcp/src/tools/causal_walk.rs` `classify_starts` — needs a new receipt/event type (`dispatch_inputs`)
- **119. Secret version id differs** (medium) — `crates/vestige-mcp/src/tools/causal_walk.rs` `classify_starts` — needs a new receipt/event type (`secret_version_id`)
- **120. Two ordered receipts, no command class** (low) — `crates/vestige-mcp/src/tools/causal_walk.rs` `git_rank` — needs a new receipt/event type (`ordered_receipt`)
- **121. Midpoint sum overflows** (high) — `crates/vestige-mcp/src/tools/repo_ingest.rs` `blame_at` — needs a new receipt/event type (`length_field`)
- **122. Inclusive end counted as exclusive** (high) — `crates/vestige-mcp/src/tools/causal_walk.rs` `git_rank` — needs a new receipt/event type (`requested_count`)
- **123. Exclusive range leaves the next value unmatched** (medium) — `crates/vestige-mcp/src/tools/causal_walk.rs` `split_frame` — needs a new receipt/event type (`diagnostic_span`)
- **124. The overflow fix is still undefined** (medium) — `crates/vestige-mcp/src/tools/causal_walk.rs` `git_rank` — needs a new receipt/event type (`line_history`)
- **125. Omitted field versus a null token** (high) — `crates/vestige-mcp/src/auto_connect.rs` `classify_marked_token` — needs a new receipt/event type (`raw_body`)
- **126. Nil pointer with a non-zero length** (high) — `crates/vestige-mcp/src/tools/repo_ingest.rs` `blame_at` — needs a new receipt/event type (`race_receipt`)
- **127. A zero stands in for a missing value** (high) — `crates/vestige-mcp/src/tools/causal_walk.rs` `classify_starts` — needs a new receipt/event type (`decode_body`)
- **128. A number decoded as text, or the reverse** (medium) — `crates/vestige-mcp/src/tools/causal_walk.rs` `git_rank` — needs a new receipt/event type (`input_output_tag`)
- **129. Two lock orders deadlock** (high) — `crates/vestige-mcp/src/tools/causal_walk.rs` `git_rank` — needs a new receipt/event type (`lock_order`)
- **130. The deadlock test returns before the second lock** (high) — `crates/vestige-mcp/src/walk_verify/run.rs` `prove` — needs a new receipt/event type (`lock_hold`)
- **131. A lock is held across the reverse acquire** (high) — `crates/vestige-mcp/src/tools/causal_walk.rs` `git_rank` — needs a new receipt/event type (`lock_hold`)
- **132. Acquire traces show opposite orders** (high) — `crates/vestige-mcp/src/tools/causal_walk.rs` `classify_starts` — needs a new receipt/event type (`lock_acquire`)
- **133. A failed decode retains memory** (medium) — `crates/vestige-mcp/src/tools/causal_walk.rs` `git_rank` — needs a new receipt/event type (`profile_count`)
- **134. Free and use name two lines** (high) — `crates/vestige-mcp/src/tools/causal_walk.rs` `split_frame` — needs a new receipt/event type (`sanitizer_sites`)
- **135. Retained bytes jump between two profiles** (medium) — `crates/vestige-mcp/src/tools/causal_walk.rs` `git_rank` — needs a new receipt/event type (`profile_count`)
- **136. Allocator id differs** (low) — `crates/vestige-mcp/src/tools/causal_walk.rs` `git_rank` — needs a new receipt/event type (`allocator_id`)
- **137. Cost jumps at a recorded size** (high) — `crates/vestige-mcp/src/walk_verify/run.rs` `prove` — needs a new receipt/event type (`size_duration`)
- **138. Time grows faster than the size** (medium) — `crates/vestige-mcp/src/tools/causal_walk.rs` `git_rank` — needs a new receipt/event type (`size_duration`)
- **139. A hardware counter moves and the blob does not** (medium) — `crates/vestige-mcp/src/tools/causal_walk.rs` `git_rank` — needs a new receipt/event type (`counter_id`)
- **140. A NaN comparison returns the wrong branch** (high) — `crates/vestige-core/src/advanced/git_records.rs` `import_target` — needs a new receipt/event type (`result_bits`)
- **141. Equal under a compare, unequal as bits** (medium) — `crates/vestige-mcp/src/tools/causal_walk.rs` `git_rank` — needs a new receipt/event type (`bit_pattern`)
- **142. Decimal bytes and parsed bits disagree across a SHA** (medium) — `crates/vestige-mcp/src/tools/causal_walk.rs` `classify_starts` — needs a new receipt/event type (`decimal_bits`)
- **143. A finite input becomes a non-finite output** (low) — `crates/vestige-mcp/src/tools/repo_ingest.rs` `blame_at` — needs a new receipt/event type (`numeric_class`)
- **144. A civil time does not exist on that day** (high) — `crates/vestige-mcp/src/tools/causal_walk.rs` `git_admissible` — needs a new receipt/event type (`zone_civil_time`)
- **145. A local duration drops or repeats an hour** (high) — `crates/vestige-mcp/src/tools/causal_walk.rs` `git_rank` — needs a new receipt/event type (`zone_duration`)
- **146. Zone data changed, the code blob did not** (high) — `crates/vestige-mcp/src/tools/causal_walk.rs` `git_rank` — needs a new receipt/event type (`zone_data_blob`)
- **147. A receipt carries second equal to 60** (medium) — `crates/vestige-mcp/src/tools/causal_walk.rs` `git_rank` — needs a new receipt/event type (`leap_second`)
- **148. Invalid input bytes grow a decoder** (high) — `crates/vestige-mcp/src/tools/causal_walk.rs` `classify_starts` — needs a new receipt/event type (`decoder_input`)
- **149. Leading bytes change a stored checksum** (medium) — `crates/vestige-mcp/src/tools/causal_walk.rs` `git_rank` — needs a new receipt/event type (`stored_checksum`)
- **150. The same bytes, two decoders** (medium) — `crates/vestige-mcp/src/tools/causal_walk.rs` `git_rank` — needs a new receipt/event type (`decoder_id`)
- **151. A decoded line and a byte length disagree** (low) — `crates/vestige-mcp/src/tools/repo_ingest.rs` `blame_at` — needs a new receipt/event type (`count_pair`)
- **152. A short write is recorded as a full write** (high) — `crates/vestige-mcp/src/walk_verify/run.rs` `prove` — needs a new receipt/event type (`short_write`)
- **153. A retry fills the unread tail with zeros** (medium) — `crates/vestige-mcp/src/tools/causal_walk.rs` `git_rank` — needs a new receipt/event type (`write_offset`)
- **154. The same idempotency key, two bodies** (high) — `crates/vestige-mcp/src/tools/causal_walk.rs` `classify_starts` — needs a new receipt/event type (`idempotency_body`)
- **155. The path checked is not the path written** (medium) — `crates/vestige-mcp/src/tools/causal_walk.rs` `classify_starts` — needs a new receipt/event type (`checked_path`)
- **156. Two tool versions, one file blob** (medium) — `crates/vestige-core/src/advanced/git_records.rs` `lock_kind` — needs a new receipt/event type (`tool_checksum`)
- **157. Applied schema id, file blob moved** (high) — `crates/vestige-mcp/src/tools/causal_walk.rs` `git_rank` — needs a new receipt/event type (`applied_schema`)
- **158. A field present on one body and absent on the other** (medium) — `crates/vestige-mcp/src/auto_connect.rs` `classify_marked_token` — needs a new receipt/event type (`body_hash`)
- **159. Two sizes for one opaque type id** (medium) — `crates/vestige-mcp/src/tools/causal_walk.rs` `git_rank` — needs a new receipt/event type (`opaque_size`)
- **160. An archive entry leaves the destination prefix** (high) — `crates/vestige-mcp/src/tools/causal_walk.rs` `classify_starts` — needs a new receipt/event type (`archive_entry`)
- **161. A receipt names a path whose blob is the secret** (medium) — `crates/vestige-mcp/src/tools/causal_walk.rs` `git_rank` — needs a new receipt/event type (`secret_blob_hash`)
- **162. The check and the use are different receipts** (high) — `crates/vestige-mcp/src/tools/causal_walk.rs` `git_rank` — needs a new receipt/event type (`check_use`)
- **163. The next hop's URL bytes are not the previous hop's** (high) — `crates/vestige-mcp/src/auto_connect.rs` `classify_marked_token` — needs a new receipt/event type (`hop_url`)
- **164. A font blob changes the pixels** (medium) — `crates/vestige-mcp/src/tools/causal_walk.rs` `paths_identify` — needs a new receipt/event type (`font_pixel_hash`)
- **165. A scale factor changes the pixel hash** (low) — `crates/vestige-mcp/src/tools/causal_walk.rs` `classify_starts` — needs a new receipt/event type (`scale_pixel_hash`)
- **166. A viewport width changes a pixel bound** (high) — `crates/vestige-mcp/src/tools/causal_walk.rs` `split_frame` — needs a new receipt/event type (`viewport_pixel`)
- **167. Tool argument bytes fail a schema hash** (high) — `crates/vestige-mcp/src/tools/causal_walk.rs` `classify_starts` — needs a new receipt/event type (`tool_schema_hash`)
- **168. Prompt body hash differs, model id does not** (high) — `crates/vestige-mcp/src/tools/causal_walk.rs` `classify_starts` — needs a new receipt/event type (`prompt_body_hash`)
- **169. A tool result is shorter than its length field** (medium) — `crates/vestige-mcp/src/tools/causal_walk.rs` `git_rank` — needs a new receipt/event type (`declared_length`)
- **170. A second call is not a replay of the first** (medium) — `crates/vestige-mcp/src/walk_verify/run.rs` `prove` — needs a new receipt/event type (`argv_bytes`)
- **173. A short sequential read is treated as the end** (high) — `crates/vestige-mcp/src/tools/causal_walk.rs` `git_rank` — needs a new receipt/event type (`short_read`)
- **174. A short pread is returned as success** (high) — `crates/vestige-mcp/src/tools/causal_walk.rs` `git_rank` — needs a new receipt/event type (`short_pread`)
- **175. One log index, two entry hashes** (high) — `crates/vestige-mcp/src/tools/causal_walk.rs` `git_rank` — needs a new receipt/event type (`log_index_hash`)
- **176. Acked index ahead of the durable index** (high) — `crates/vestige-mcp/src/tools/causal_walk.rs` `classify_starts` — needs a new receipt/event type (`acked_index`)
- **177. A log index is gone after compaction** (high) — `crates/vestige-mcp/src/tools/causal_walk.rs` `git_rank` — needs a new receipt/event type (`compaction_range`)
- **178. One locale id, two sort orders** (high) — `crates/vestige-mcp/src/tools/causal_walk.rs` `git_rank` — needs a new receipt/event type (`collator_id`)
- **179. A collator write passes the buffer** (high) — `crates/vestige-mcp/src/tools/causal_walk.rs` `classify_starts` — needs a new receipt/event type (`collator_counts`)
- **180. Punctuation weight is never consulted** (medium) — `crates/vestige-mcp/src/tools/causal_walk.rs` `git_rank` — needs a new receipt/event type (`collation_strength`)
- **181. Gzip timestamp, same source tree** (high) — `crates/vestige-mcp/src/tools/causal_walk.rs` `git_rank` — needs a new receipt/event type (`archive_timestamp`)
- **182. ZIP entry time, same member bytes** (medium) — `crates/vestige-mcp/src/tools/causal_walk.rs` `git_rank` — needs a new receipt/event type (`archive_timestamp`)
- **183. Intent URL present, initial URL empty** (high) — `crates/vestige-mcp/src/tools/causal_walk.rs` `classify_starts` — needs a new receipt/event type (`intent_url`)
- **184. Web process died, URL event absent** (high) — `crates/vestige-mcp/src/tools/causal_walk.rs` `git_rank` — needs a new receipt/event type (`process_exit`)
- **185. Launcher relaunch restores the last route** (high) — `crates/vestige-mcp/src/tools/causal_walk.rs` `classify_starts` — needs a new receipt/event type (`restored_state`)
- **186. Focus lands on the wrong control** (high) — `crates/vestige-mcp/src/tools/causal_walk.rs` `git_rank` — needs a new receipt/event type (`focus_id`)
- **187. Focus moves while the node stays** (medium) — `crates/vestige-mcp/src/tools/causal_walk.rs` `walk_payload` — needs a new receipt/event type (`focus_presence`)
- **188. Reading order skips the opened container** (high) — `crates/vestige-mcp/src/tools/causal_walk.rs` `paths_identify` — needs a new receipt/event type (`a11y_order`)
- **189. Start edge follows flex direction** (high) — `crates/vestige-mcp/src/tools/causal_walk.rs` `git_rank` — needs a new receipt/event type (`writing_direction`)
- **190. Logical order and visual order disagree** (medium) — `crates/vestige-mcp/src/tools/causal_walk.rs` `classify_starts` — needs a new receipt/event type (`visual_order`)
- **191. A date control rejects a longer value** (medium) — `crates/vestige-mcp/src/tools/causal_walk.rs` `git_rank` — needs a new receipt/event type (`control_value`)
- **192. Two engines, one stylesheet, two used values** (medium) — `crates/vestige-mcp/src/walk_verify/run.rs` `prove` — needs a new receipt/event type (`engine_used_value`)
- **193. A resumed session skips the chain check** (high) — `crates/vestige-mcp/src/tools/causal_walk.rs` `git_rank` — needs a new receipt/event type (`session_resume`)
- **194. Name bytes fail a hostname check** (medium) — `crates/vestige-mcp/src/tools/causal_walk.rs` `classify_starts` — needs a new receipt/event type (`certificate_name`)
- **195. One key, one nonce, two ciphertexts** (high) — `crates/vestige-mcp/src/tools/causal_walk.rs` `git_rank` — needs a new receipt/event type (`key_nonce`)
- **196. The certificate's end time is already past** (medium) — `crates/vestige-mcp/src/tools/causal_walk.rs` `git_admissible` — needs a new receipt/event type (`certificate_not_after`)
- **197. Expected audience, no audience field, accepted** (high) — `crates/vestige-mcp/src/tools/causal_walk.rs` `classify_starts` — needs a new receipt/event type (`audience_present`)
- **198. A use with no check receipt** (high) — `crates/vestige-mcp/src/tools/causal_walk.rs` `classify_starts` — needs a new receipt/event type (`check_use`)
- **199. Detected license id differs from the manifest id** (high) — `crates/vestige-mcp/src/auto_connect.rs` `classify_marked_token` — needs a new receipt/event type (`license_id`)
- **200. The same license bytes, two checker verdicts** (medium) — `crates/vestige-mcp/src/tools/causal_walk.rs` `classify_starts` — needs a new receipt/event type (`license_checker`)
- **201. The license file moved, the id did not** (medium) — `crates/vestige-core/src/advanced/git_records.rs` `lock_kind` — needs a new receipt/event type (`license_id`)
- **202. Stored checksum differs from the recomputed one** (high) — `crates/vestige-mcp/src/tools/causal_walk.rs` `git_rank` — needs a new receipt/event type (`checksum_pair`)
- **203. Two replicas, one key, two value hashes** (high) — `crates/vestige-mcp/src/tools/causal_walk.rs` `apply_version_range` — needs a new receipt/event type (`replica_value_hash`)
- **204. Page length and page checksum disagree** (medium) — `crates/vestige-mcp/src/tools/repo_ingest.rs` `blame_at` — needs a new receipt/event type (`page_checksum`)
- **205. A negative cache entry is served past its deadline** (medium) — `crates/vestige-mcp/src/tools/causal_walk.rs` `git_rank` — needs a new receipt/event type (`negative_cache`)
- **206. One cache key, two body hashes** (high) — `crates/vestige-mcp/src/tools/causal_walk.rs` `classify_starts` — needs a new receipt/event type (`cache_body_hash`)
- **207. An offset is committed while records are in flight** (high) — `crates/vestige-mcp/src/tools/causal_walk.rs` `git_rank` — needs a new receipt/event type (`committed_offset`)
- **208. Consumer sequence numbers go backwards** (medium) — `crates/vestige-mcp/src/walk_verify/run.rs` `prove` — needs a new receipt/event type (`consumer_sequence`)
- **209. Same seed, different file order, different weights** (high) — `crates/vestige-mcp/src/tools/causal_walk.rs` `git_rank` — needs a new receipt/event type (`seed_order_hash`)
- **210. Same features, a different label histogram** (medium) — `crates/vestige-mcp/src/tools/causal_walk.rs` `walk_payload` — needs a new receipt/event type (`label_histogram`)
- **212. PE TimeDateStamp differs across otherwise identical links** (high) — `crates/vestige-mcp/src/tools/causal_walk.rs` `git_rank` — needs a new receipt/event type (`link_timestamp`)
- **213. TimeDateStamp is a hash read as a clock** (medium) — `crates/vestige-mcp/src/tools/causal_walk.rs` `git_rank` — needs a new receipt/event type (`field_kind`)
- **214. Two image exports, same epoch, different mountpoint mtimes** (high) — `crates/vestige-mcp/src/tools/causal_walk.rs` `git_rank` — needs a new receipt/event type (`export_mtime`)
- **215. An export names a digest the store does not have** (high) — `crates/vestige-mcp/src/tools/causal_walk.rs` `git_rank` — needs a new receipt/event type (`missing_digest`)
- **216. Language-id bytes miss the plural table** (high) — `crates/vestige-mcp/src/tools/causal_walk.rs` `classify_starts` — needs a new receipt/event type (`plural_rule`)
- **217. Punycode input and decoded name differ** (high) — `crates/vestige-mcp/src/tools/causal_walk.rs` `git_rank` — needs a new receipt/event type (`punycode_bytes`)
- **218. Decoder accepts non-ASCII bytes inside an ACE label** (medium) — `crates/vestige-mcp/src/walk_verify/run.rs` `prove` — needs a new receipt/event type (`ace_label`)
- **219. Permission revoked, process exits, no change receipt** (high) — `crates/vestige-mcp/src/tools/causal_walk.rs` `classify_starts` — needs a new receipt/event type (`permission_id`)
- **220. One-time permission expires while another activity is in front** (high) — `crates/vestige-mcp/src/tools/causal_walk.rs` `git_rank` — needs a new receipt/event type (`permission_id`)
- **221. Escape is delivered and the layer stays open** (high) — `crates/vestige-mcp/src/tools/causal_walk.rs` `git_rank` — needs a new receipt/event type (`key_event`)
- **222. One Escape closes two layers** (high) — `crates/vestige-mcp/src/tools/causal_walk.rs` `git_rank` — needs a new receipt/event type (`layer_id`)
- **223. File size is not a multiple of the page size** (high) — `crates/vestige-mcp/src/tools/causal_walk.rs` `git_rank` — needs a new receipt/event type (`page_size`)
- **224. A WAL record names a page that is uninitialized** (high) — `crates/vestige-mcp/src/tools/causal_walk.rs` `git_rank` — needs a new receipt/event type (`wal_page`)
- **225. Fit stored column ids that the next array does not** (high) — `crates/vestige-mcp/src/tools/causal_walk.rs` `git_rank` — needs a new receipt/event type (`column_id_list`)
- **226. Accessible name is empty and a description id is set** (high) — `crates/vestige-mcp/src/tools/causal_walk.rs` `git_rank` — needs a new receipt/event type (`accessible_name`)
- **227. Contrast status is needs-review because the background is not one color** (medium) — `crates/vestige-mcp/src/tools/causal_walk.rs` `git_rank` — needs a new receipt/event type (`contrast_status`)
- **228. A killed worker restarts later than its recorded delay** (medium) — `crates/vestige-mcp/src/tools/causal_walk.rs` `git_rank` — needs a new receipt/event type (`restart_delay`)
- **229. A file lock returns deadlock with no second holder** (high) — `crates/vestige-mcp/src/tools/causal_walk.rs` `git_rank` — needs a new receipt/event type (`file_lock`)
- **230. A lock wait outlives the holder** (high) — `crates/vestige-mcp/src/tools/causal_walk.rs` `git_rank` — needs a new receipt/event type (`lock_pid`)
- **231. A preflight header is on the first hop and missing on the next** (high) — `crates/vestige-mcp/src/tools/causal_walk.rs` `git_rank` — needs a new receipt/event type (`hop_header`)
- **232. A response names a header the request did not carry** (medium) — `crates/vestige-mcp/src/tools/causal_walk.rs` `classify_starts` — needs a new receipt/event type (`response_header`)
- **233. The container metric is under the limit and the cgroup max is not** (high) — `crates/vestige-mcp/src/tools/causal_walk.rs` `git_rank` — needs a new receipt/event type (`cgroup_max`)
- **234. Usage is over the limit and nothing is killable** (high) — `crates/vestige-mcp/src/tools/causal_walk.rs` `classify_starts` — needs a new receipt/event type (`cgroup_kill`)
- **235. Reclaim runs at the limit and the kill count stays zero** (high) — `crates/vestige-mcp/src/tools/causal_walk.rs` `classify_starts` — needs a new receipt/event type (`oom_kill_count`)
- **236. The worker script installed from cache is not the network body** (high) — `crates/vestige-mcp/src/tools/causal_walk.rs` `git_rank` — needs a new receipt/event type (`worker_body_hash`)
- **237. A wall-clock delta and a monotonic delta disagree** (high) — `crates/vestige-mcp/src/tools/causal_walk.rs` `git_rank` — needs a new receipt/event type (`clock_delta`)
- **238. The requested name is absent from the certificate's name bytes** (high) — `crates/vestige-mcp/src/tools/causal_walk.rs` `git_rank` — needs a new receipt/event type (`certificate_san`)
- **239. No subject-alt bytes, and the common-name bytes are rejected** (medium) — `crates/vestige-mcp/src/tools/causal_walk.rs` `classify_starts` — needs a new receipt/event type (`certificate_cn`)

