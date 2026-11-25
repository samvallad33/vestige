## AI Agent & Instruction-File Security Scope

Prompt-injection payloads concealed inside `AGENTS.md`, `CLAUDE.md`, `.cursorrules`, `.cursor/rules/`, or pull request diffs (including invisible Unicode TAG characters, Trojan Source bidirectional overrides, and unauthorized Biba control-plane workflow edits) are explicitly in scope for security reports.

This repository enforces deterministic pre-merge inspection using Vestige Shield (`vestige shield --sarif --fail-on-finding`) and `merge=vestige-guard`. To reproduce or verify a report locally without cloud access, run `vestige redteam` and `vestige doctor --context-rot`.
