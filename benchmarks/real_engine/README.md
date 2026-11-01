# Real-engine benchmark campaign

The campaign compares four arms: the bare model, BM25 memory, pinned local
BGE-small retrieval, and **one Vestige arm with all applicable hooks**. Every
solver and semantic judge uses `gpt-5.4-nano`; there is no local LLM fallback.
The actual state is in `runs/RESULTS.md`, each benchmark's `RESULTS.md`, and the
usage-based SQLite ledger. Source inventories and engine self-tests are setup
evidence and do not supply benchmark scores.

The Vestige arm uses the repository-built `target/release/vestige-mcp` child over
stdio JSON-RPC through `benchmarks/memconflict/mcp_client.VestigeMCP`. The build
manifest records its SHA-256 and committed source. A fresh independent task or
conversation owns its engine, Strata directory, policy history, home, temporary
files and hook state. A missing or stopped engine fails the item.

Memory-QA bare receives the question alone and has no long-term memory store.
Interactive and security benchmarks give every arm the same official visible
current-session task inputs; their bare arm retains the official task tools.
System prompts otherwise differ only by a neutral tools-description line.
Gold targets, scoring annotations and answer text stay outside solver inputs,
memory tags, summaries and retrieval construction.

## Reproduction

The shell entry point fetches official public sources at pinned commits, checks
dataset bytes, installs the recorded Python dependencies, builds the release
binary, writes its manifest, freezes the full66-task Terminal-Bench catalog, and launches the global campaign. It never launches
a local LLM. Python3.13 is needed by the official AutomationBench package; a new
environment uses that version. An existing environment is preserved and may
leave AutomationBench blocked until its Python requirement is satisfied.

```sh
bash benchmarks/real_engine/reproduce.sh --prepare-only
bash benchmarks/real_engine/reproduce.sh
bash benchmarks/real_engine/reproduce.sh --benchmarks locomo,longmemeval,mpbench
bash benchmarks/real_engine/reproduce.sh --report-only
```

The paid run reads an existing owner-only `runs/.env.openai` file containing an
`OPENAI_API_KEY` assignment. Credentials are never printed or committed. The
`--prepare-only` and `--report-only` routes make no model requests. Optional
`--runs PATH` selects a separate evidence directory. An upstream checkout at a
different revision or with tracked local changes is preserved and causes a
setup failure instead of being reset.

Each generated `runs/<benchmark>/reproduce.sh` delegates to this entry point.
Use one campaign process. The campaign lock refuses a second process, allowing
concurrent benchmarks to share one40-request asyncio limit and `runs/budget.db`.
Memory benchmarks use ten deterministic workers per arm; long agent benchmarks
use five. Benchmarks may execute concurrently under the same global40-request
limit. Shards use the full SHA-256 of the UTF-8 item identifier modulo the worker count;
the recorded seed controls sampling/trials rather than Python's randomized hash.

The dependency snapshot is in `requirements.lock`. Local editable projects are
excluded from index dependencies: AgentDojo is installed from its pinned public
checkout. Preserve the Python version and platform alongside a completed run;
package versions alone do not guarantee identical native wheels on another OS.

## Spend, checkpoints and scoring

Every API attempt first reserves a conservative input/output cost bound in the
one process-safe SQLite ledger. Returned usage reconciles the reservation.
Ambiguous failed attempts remain encumbered until provider evidence establishes
their charge; a retry never silently releases unknown spend. At most six
retries follow429/5xx responses, with exponential backoff and jitter. The total
campaign limit is$25, including judges, hooks and unsuccessful attempts.

A one-item calibration prints each benchmark's projected cost before its full
workers start. Projections are forecasts only: they do not skip a full set while
actual bounded calls remain affordable. Every benchmark uses its full official
set, with no budget-based sampling. Terminal-Bench uses all66 official tasks and
three trials per task/arm, or792 total rollouts across four arms. The earlier
20-task manifest is preserved as superseded preparation. Only the actual shared
$25 reservation gate stops further API work, with exact finished and unfinished
item identities retained in checkpoints; the requested denominator never shrinks.

Each completed item is immediately appended and fsynced to its worker JSONL.
Restarts reuse only completed items with the same input/model/binary/prompt
identity. Failures retain their raw logs and retry at the end. Missing or
duplicate official item/trial identities prevent merged scoring. Official data,
splits and scorer algorithms are unchanged. MPBench is the sole owner-authorized
exception: its reviewed paper-metric scorer was committed locally before any
arm, because upstream provides data but no scorer.

The reporter invokes the independent `integrity.audit(repo, runs)` before any
new `RESULTS.md` write. The audit records its evidence in `runs/INTEGRITY.md`
and `runs/integrity.json`. A failed benchmark becomes **INVALID** and is excluded;
missing evidence remains pending. The audit requires an unscored qualification
sample of20 fixed-seed random official items per arm and at least five observed samples per applicable hook.
This qualification collects raw evidence only, with no official metric or judge
scoring. Its checkpoint identity and complete official manifest are retained for
the later full-set continuation. All `RESULTS.md` writes stay held until the
independent audit returns report readiness; STATUS and INTEGRITY carry interim
diagnostics.
The renderer never edits source or raw output to conceal an integrity finding.

The reporter never estimates a missing benchmark result. It accepts recorded
metrics only when complete scored artifacts have qualified identities explicitly
approved by a PASS result for both the benchmark and arm. Current
qualification metadata belongs in `runs/QUALIFICATION.json`:

```json
{
  "status": "qualified",
  "vestige_arm": "all_applicable_hooks",
  "hooks_manifest_sha256": "SHA256_OF_AUDITED_HOOK_MANIFEST",
  "approved_run_identities": ["CHECKPOINT_RUN_IDENTITY_SHA256"]
}
```

The independent audit must also return each eligible identity in its
`approved_run_identities` mapping; a matching hooks hash alone is insufficient.
This marker records an independently checked state; writing the marker alone
does not establish qualification. Old engine-only
pilots cannot become full-hook scores by renaming a folder. Their spend remains
in the ledger and their output counts remain visible as superseded preparation.

Call `reporting.generate(repo, runs)` after a meaningful milestone or at the end
of a campaign. It creates one results report and reproduction script for each
of the fourteen benchmark entries, plus the summary table. `runs/STATUS.md`
updates every five minutes during execution. Reports contain means, supplied
bootstrap intervals, scored n, trials/seeds, model, actual spend, pending
reservations, official data identities and engine hash, or an explicit pending
or skipped state.

## Interpretation limits

All earlier task-guided `session-window-debug` results are invalid. Historical
`proof.json` files are engine self-tests, never Terminal-Bench pass-rate evidence.
GraphWalks' full-context comparison uses the same context limit as its actual
graph-memory route. CWE-bench v1 is skipped because official tasks are private;
its training corpus is not a substitute, and any future result is secondary
security evidence. Vals Finance Agent v2 is skipped because its official scorer
and held-out tasks are not publicly available.

MPBench's public files contain3,241 attacks and2,999 benign records, differing
from the paper's stated counts. All objects are preserved; two missing targets
remain unscorable. ASR judges actual persisted entries, rather than treating all
admissions as attacks. Benign FPR requires an observed defense decision and is
unavailable when no detector is installed or no decision was observed. The raw
skill artifact is delivered, while an ASR-only memory assay does not establish
autonomous skill synthesis or cross-session retrieval success.

Published Mem0, Zep and Letta figures appear only in a separate column labelled
**published, different model or judge**. `published.json` records primary sources,
models, judges, question-scope differences and publisher caveats. Those numbers
do not supply a head-to-head comparison, a combined ranking, or missing control
results. Lower Vestige outcomes must be reported as directly as higher outcomes.

Nothing in this campaign pushes to GitHub, deploys production, changes billing,
or substitutes public training data for private official evaluations. Commits
remain local to `bench/real-engine`.
