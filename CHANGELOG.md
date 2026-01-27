# Changelog

All notable changes to Vestige will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.1.2] - 2025-01-27

### Fixed
- Embedding model cache now uses platform-appropriate directories instead of polluting project folders
  - macOS: `~/Library/Caches/com.vestige.core/fastembed`
  - Linux: `~/.cache/vestige/fastembed`
  - Windows: `%LOCALAPPDATA%\vestige\cache\fastembed`
- Can still override with `FASTEMBED_CACHE_PATH` environment variable

---

## [1.1.1] - 2025-01-27

### Fixed
- UTF-8 string slicing issues in keyword search and prospective memory
- Silent error handling in MCP stdio protocol
- Feature flag forwarding between crates
- All GitHub issues resolved (#1, #3, #4)

### Added
- Pre-built binaries for Linux, Windows, and macOS (Intel & ARM)
- GitHub Actions CI/CD for automated releases

---

## [1.1.0] - 2025-01-26

### Changed
- **Tool Consolidation**: 29 tools → 8 cognitive primitives
  - `recall`, `semantic_search`, `hybrid_search` → `search`
  - `get_knowledge`, `delete_knowledge`, `get_memory_state` → `memory`
  - `remember_pattern`, `remember_decision`, `get_codebase_context` → `codebase`
  - 5 intention tools → `intention`
- Stats and maintenance moved from MCP to CLI (`vestige stats`, `vestige health`, etc.)

### Added
- CLI admin commands: `vestige stats`, `vestige health`, `vestige consolidate`, `vestige restore`
- Feedback tools: `promote_memory`, `demote_memory`
- 30+ FAQ entries with verified neuroscience claims
- Storage modes documentation: Global, per-project, multi-Claude household
- CLAUDE.md templates for proactive memory use
- Version pinning via git tags

### Deprecated
- Old tool names (still work with warnings, removed in v2.0)

---

## [1.0.0] - 2025-01-25

### Added
- FSRS-6 spaced repetition algorithm with 21 parameters
- Bjork & Bjork dual-strength memory model (storage + retrieval strength)
- Local semantic embeddings with fastembed v5 (BGE-base-en-v1.5, 768 dimensions)
- HNSW vector search with USearch (20x faster than FAISS)
- Hybrid search combining BM25 keyword + semantic + RRF fusion
- Two-stage retrieval with reranking (+15-20% precision)
- MCP server for Claude Desktop integration
- Tauri desktop application
- Codebase memory module for AI code understanding
- Neuroscience-inspired memory mechanisms:
  - Synaptic Tagging and Capture (retroactive importance)
  - Context-Dependent Memory (Tulving encoding specificity)
  - Spreading Activation Networks
  - Memory States (Active/Dormant/Silent/Unavailable)
  - Multi-channel Importance Signals (Novelty/Arousal/Reward/Attention)
  - Hippocampal Indexing (Teyler & Rudy 2007)
- Prospective memory (intentions and reminders)
- Sleep consolidation with 5-stage processing
- Memory compression for long-term storage
- Cross-project learning for universal patterns

### Changed
- Upgraded embedding model from all-MiniLM-L6-v2 (384d) to BGE-base-en-v1.5 (768d)
- Upgraded fastembed from v4 to v5

### Fixed
- SQL injection protection in FTS5 queries
- Infinite loop prevention in file watcher
- SIGSEGV crash in vector index (reserve before add)
- Memory safety with Mutex wrapper for embedding model

## [0.1.0] - 2026-01-24

### Added
- Initial release
- Core memory storage with SQLite + FTS5
- Basic FSRS scheduling
- MCP protocol support
- Desktop app skeleton
