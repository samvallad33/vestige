# v3 candidate validation

The release candidate has three separate gates: local deterministic correctness,
real-binary integration, and target-platform CI. A provider-cost claim additionally
requires the frozen empirical evaluation described in `V3-ROADMAP.md`.

## Core and protocol

```sh
CARGO_INCREMENTAL=0 cargo test --workspace --no-fail-fast
CARGO_INCREMENTAL=0 cargo clippy --workspace --all-targets -- -D warnings
CARGO_INCREMENTAL=0 cargo test -p vestige-mcp --test e2e_real_binary -- --ignored --test-threads=1
python3 scripts/test-tool-frontier.py --output tool-frontier.json
python3 scripts/test-context-evidence.py
```

The optional real-embedding suite requires a local embedding artifact/runtime.
It checks semantic retrieval, corrected ingestion, contradiction preservation,
restart persistence, corrupt-FTS recovery and embedding purge. Its ignored marker
is a dependency choice, not an expected failing assertion. Other optional
artifacts (such as Granite) and target-platform tests remain separate gates.

The v3 core regression module adds deterministic concurrent-writer, restart,
transaction-failure, namespace/protection and purge cases. A cascade journal
failure must roll back the neighbor penalty; a lifecycle page failure must roll
back earlier row updates. Two stores racing a cascade must produce one effect.
GC must resume past rows it deleted, and delayed embedding work must never
resurrect a purged memory. Dashboard replay regressions verify exact persisted
evidence identity, empty results, invalid run IDs and injected receipt-write
failure. Witness regressions reject prose as memory identity, preserve real path
order and prevent fabricated edges across prose gaps.

## Runtime and evaluation

```sh
python3 -m pip install ./integrations/python
python3 -m unittest discover -s integrations/python/tests -v
python3 -m unittest discover -s examples/python/tests -v
python3 -m unittest discover -s benchmarks/task-cost/tests -v
python3 benchmarks/task-cost/developer_suite.py --self-test
python3 scripts/test-runtime-integration.py --binary target/debug/vestige-mcp
```

Runtime cases cover explicit packet acknowledgment, compaction, invalid call
identities, malformed unchanged responses, failed tools, ownership against caller
mutation, and a deterministic 100-step retention/revision sequence. Transport
cases cover stalled initialization, a nonreading child, response limits and cleanup.

Evaluation cases preserve failed/timed-out drivers and prohibit scoring them as
success. They reject modified drivers/time policies, duplicate request identities,
nonfinite charges and incomplete evidence. No model agent or paid request runs
in these tests. The v3 CI workflow runs Python contracts on Python 3.10 and 3.12
and qualifies the installed package over real MCP stdio on Linux.

## Dashboard and hooks

```sh
pnpm --filter @vestige/dashboard check
pnpm --filter @vestige/dashboard test
pnpm --filter @vestige/dashboard build
python3 -m unittest discover -s tests/hooks -p 'test_*.py'
python3 scripts/run-dashboard-e2e.py --log-dir ./browser-test-evidence
```

Run the browser command after the frontend build/check finish: Vite development
and production builds share generated paths. Use a supported Node LTS runtime.
The runner pins a copy of the executable, creates a disposable database, assigns
fresh loopback ports, validates
MCP initialization and reads its seeded data back before starting Playwright.
Synthetic graph topology is explicit and independent of embedding readiness.
Credentials for this fixture are generated in memory; tests never load a token
from the user's home directory. The runner closes its owned processes and store.

Pass Playwright file names or selectors after the runner options for targeted
reproduction. Screenshots, videos and traces remain in the dashboard test output.
Failures remain failures; unavailable artifacts, optional tests and platform checks
must be reported separately rather than counted as passing.

The browser suite includes a small number of pre-existing quarantined tests for
retired Graph3D rainbow-burst and Pulse Toast surfaces. They are reported as
skipped, never as passing. Current Witness, replay, explicit Promote/Walk and
settings maintenance flows have active replacements. Pixel coverage varies with
corpus size; the small fixture's smoke gate checks luminance, spatial variation
and motion, and records coverage rather than reusing a thousand-memory visual
baseline. Frozen-frame checks compare frame changes to measured reload noise.

The browser runner records the tested binary digest and fixture assertions in
its log directory. Use its exit status alongside the Playwright report; the
presence of screenshots alone is not a passing result.

## Automatic memory writes

The default review mode is `fast`: memory writes apply immediately without a
Memory PR approval step, including preferences and other content that optional
risk classification flags. Receipts remain available. `risk_gated` and `paranoid`
are explicit opt-ins, selectable in the Memory PR dashboard. Existing valid
settings survive upgrades; users can switch to Automatic without approving each
future write. This does not apply historical pending proposals or remove the
explicit `confirm=true` contract for purge.

Missing or malformed mode files use the automatic default (malformed settings
are logged). The settings API rejects invalid mode names without changing the
current setting. Tests cover fresh-store ingestion and retrieval after restart,
opt-in review, unchanged historical proposals, invalid settings, failed UI saves,
persisted mode changes and the automatic empty state.

Clean-runner CI exposed an embedding-profile activation regression: migrated
vectors could be excluded by the legacy availability flag. Activation now updates
retrieval eligibility in the same transaction as the active-profile pointer. The
regression fixture explicitly clears legacy availability before activation, so a
developer model cache cannot hide the failure.
