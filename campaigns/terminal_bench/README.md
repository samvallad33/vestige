# Terminal-Bench4.0 CPU campaign

Official utility tasks are pinned to Terminal-Bench4.0.0 (`452bf305c6daa62fc59061d22133a7cbc7c1572e`). Harbor is pinned to `5eb713ff9500daced4e143b98765a7c732744a1b` (0.25.0). Both public repositories use Apache2.0. The historical2.1 release is a different task/version population and its scores must remain separate.

Official sources: [benchmark versions](https://www.tbench.ai/benchmarks), [4.0 run instructions](https://www.tbench.ai/run), [4.0 release](https://github.com/harbor-framework/terminal-bench/releases/tag/v4.0.0), [Harbor custom agents](https://docs.harborframework.com/agents/custom-agents). Related Vestige planning categories are [#595](https://github.com/samvallad33/vestige/issues/595) and [#596](https://github.com/samvallad33/vestige/issues/596).

The repository has66 official task definitions. Static declared-resource filtering identifies CPU-only single-container candidates within4CPUs and4GiB. These are candidates, pending image-architecture/build checks and oracle qualification. The full4.0 suite includes GPU tasks and larger memory requirements. The existing local Colima VM is arm64,4CPUs,6GiB,30GBdisk; the host has64GiB RAM. This preparation does not change task resources to make the full suite fit.

## What the adapter actually mediates

`adapter.VestigeTerminalAgent` implements Harbor's `BaseAgent` interface. It forwards complete `Bash` tool calls through the native Operator synchronous hook and Reflex call/digest acknowledgment **before** calling the original Harbor `environment.exec`. The model receives the loaded native MCP schemas and can make dynamic exact-handle memory calls. State/effect metadata and milestone backfill use native Strata receipts. The solver receives no static memory-body prompt header.

This is a custom host, labeled `benchmark-custom`. It is not an OpenCode session. It does not interdict every Linux syscall, every nested subprocess inside an admitted command, or commands from unrelated installed agents. Interactive tmux key fragments are not the execution interface in this track. Harbor's original task verifier stays outside solver authority. Command success alone is never an official task pass.

`native_vendor` is an isolated adaptation of the reviewed AgentDojo native client at local commit75cea067. It targets installed native6.2.0 while leaving the original experiments unchanged. Its provenance and exact upstream/local hashes are recorded in `native_vendor/PROVENANCE.json`.

## Runtime

The isolated Python runtime is `upstream/harbor/.venv` (Python3.13.9). Harbor's frozen lock was installed using a private uv0.13.5 tool because globaluv0.12.24 was too old. No global uv upgrade occurred. The declared AgentDojo base dependencies were added to support its read-only guarded-loop imports. Exact installed versions are recorded in `runtime.json`; no model weights were downloaded. Include explicit source paths because this host does not reliably process editable-package `.pth` files:

```sh
export PYTHONPATH=.:upstream/harbor/src:upstream/agentdojo-library/src
```

Run from the campaign workspace root. Preparation and checking make no provider calls:

```sh
upstream/harbor/.venv/bin/python -m campaigns.terminal_bench.prepare --upstream upstream/terminal-bench --output campaigns/terminal_bench/evidence/frozen --limit 10
upstream/harbor/.venv/bin/python -m campaigns.terminal_bench.prepare check campaigns/terminal_bench/evidence/frozen/manifest.json
upstream/harbor/.venv/bin/python -m pytest -q tests/test_terminal_bench.py
```

The manifest always records every task's hashes/resources; the limit caps only the candidate selection. Preparation never overwrites a freeze.

## Native treatments and qualification

`NativeStack(root,timeout=90,profile="production")` copies the current owner commitments, validated law cache, mode and STRICT into fresh trial storage. No old receipts, permits or runtime histories are copied. Installed sources and both source/copy policies are pinned; transport, missing acknowledgment or digest faults hold the run.

`profile="all_rules"` is a separate benchmark treatment. It loads the source-audited profile at `evidence/native-audit/operator-all-rules-profile.json`, applies118 exact native rule overrides to `enforce` only in the disposable policy copy, and sets native strict/test-integrity/turnover/brake variables for that process. Fifteen production SHADOW defaults are promoted. Every owner law/cache and all other policy keys remain. Global configuration is never changed.118 configured rules is not118 exercised rules; report per-rule paired fixtures separately.

```sh
upstream/harbor/.venv/bin/python -m campaigns.terminal_bench.native_vendor.qualify_native --profile all_rules
upstream/harbor/.venv/bin/python -m campaigns.terminal_bench.smoke --profile all_rules
```

The smoke uses the existing pinned local Ubuntu image,1CPU/256MiB,networknone,read-onlyroot,16MiBtmpfs,noprivileges,nohostmounts. It executes one fixed benign write/read through Harbor's original `DockerEnvironment.exec` with a synthetic bounded Docker transport. The denied payload is submitted only to native classification in a separate disposable session; it has no executor. The container is removed afterward. These are synthetic security/utility fixtures and never enter official Terminal-Bench scoring. Failed evidence remains local.

## Exact scored-run boundary

The adapter has no default API client, key discovery, or automatic model request. Missing solver setup fails with `LAUNCH_GATED`. Paid Responses solvers are explicitly gated because this adapter has no paid permit/ledger binding. The supported free route is exactly `local_solver:factory`, validated by `validate_local_solver` for the exact local wrapper, model, budget, loopback endpoint and pinned model digest. The adapter rejects unsupported factories before construction. Scripted fixture factories are only for local qualification. A future authorized free-local scored launch uses:

```sh
upstream/harbor/.venv/bin/harbor run -p <pinned_CPU_task_path> -a campaigns.terminal_bench.adapter:VestigeTerminalAgent --model qwen3.6:35b-a3b-q8_0 --ak skip_model_parsing=true --ak solver_factory=local_solver:factory --ak native_profile=all_rules --n-concurrent 1 -e docker
```

Before this command is considered ready, select the exact task bytes, qualify its unchanged oracle/verifier and image architecture, qualify the free-local solver contract and compute bounds, and freeze both control/treatment solver settings. No paid solver factory is accepted by this setup. No official Terminal-Bench score has been produced. Preparation never starts local model inference.

Rollback is removal of these isolated campaign outputs/runtime/container fixtures only. Do not delete original experiments or change installed Operator state. Preserve proof artifacts when revising the candidate.
