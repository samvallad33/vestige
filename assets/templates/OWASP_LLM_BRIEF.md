# Deterministic AI Coding-Agent Risk Mitigation Brief (OWASP LLM01 & LLM06)

This one-page summary documents how **Vestige** mitigates the two highest-severity coding-agent risk classes in the OWASP Top 10 for LLM Applications using deterministic, offline Rust verification instead of probabilistic LLM filters.

## OWASP LLM01: Prompt Injection & Concealed Code Smuggling

Coding agents consume repository instruction files (`AGENTS.md`, `CLAUDE.md`, `.cursorrules`), pull request diffs, and comments directly into privileged execution contexts. Attackers hide stealth instructions inside invisible Unicode TAG-block codepoints (`U+E0000..U+E007F`), Trojan Source bidirectional overrides (`U+202A..U+202E`, `U+2066..U+2069`), and zero-width characters.

Vestige enforces deterministic blocking at three independent layers:
1. **Git Merge Driver (`merge=vestige-guard`)** rejects concealed Unicode and hard-block secrets before `git merge` completes.
2. **Pull Request Shield (`vestige shield --sarif --fail-on-finding`)** scans every added diff line in CI, uploads native SARIF 2.1.0 annotations to GitHub Code Scanning, and fails the merge gate with exit code 2.
3. **Memory Firewall (`remember_unified`)** blocks concealed Unicode payloads from entering the Strata memory ledger, verified 50/50 by `vestige redteam`.

## OWASP LLM06: Excessive Agency & Unbounded Autonomous Loops

Autonomous coding agents operating with shell and filesystem tools can drift outside approved paths, re-introduce reverted vulnerabilities, modify CI workflows, or spin in expensive retry loops.

Vestige enforces fail-closed runtime boundaries without calling any external model:
1. **Execution Lease (`vestige lease` & `vestige steer`)** hooks directly into `PreToolUse` and `PostToolBatch` to enforce file-scope allowlists/denylists, forbidden command gates, cross-session heavy-command mutexes, spin-loop circuit breakers, and immediate loop termination (`continue: false`).
2. **Biba Integrity Boundary** flags every modification to `.github/workflows/`, `.git/hooks/`, `AGENTS.md`, `CLAUDE.md`, and `.cursorrules` for mandatory human owner review.
3. **Append-Only Cryptographic Audit Trail (`vestige compliance --eu-ai-act / --soc2`)** chains every Operator stop, human permit, and commit receipt with BLAKE3 and Ed25519 locally on disk, never uploading proprietary source code to external clouds.
