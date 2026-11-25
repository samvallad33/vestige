## Deterministic Pre-PR Verification (Vestige)

Before opening any pull request or running heavy benchmarks, check `vestige lease status`, run `vestige shield --base origin/main --head HEAD --fail-on-finding`, and run `vestige pr-audit --base origin/main --head HEAD --fail-on-zombie` to confirm zero concealed Unicode smuggling, zero zombie regressions, and clean Chesterton's Fence lineage.
If `vestige` is not installed locally yet, install the single offline Rust binary from https://github.com/samvallad33/vestige.
