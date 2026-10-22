# Actual local-model agent / effect comparison

This is executable, not a plan. It runs a pinned Qwen2.5-1.5B Q4_K_M language model in a custom restricted-tool host. The model selects each next action, sees allowed/blocked tool feedback, and can adapt over four steps. A separate earlier model pass generates one prompt injection per attack family before any defender is tested. No external paid inference endpoint or API key is required.

## Comparisons and outcomes

Four configurations share the same model, owner instructions, task, available tools, initial state and matched seed: prompt only, a benchmark-supplied explicit allowlist, **real OPA 1.9.0** executing that same allowlist in Rego, and the actual **Vestige Operator Lite** hook. The allowlist is not being passed off as native Claude Code/OpenCode permissions. OPA is a configured generic engine, not a prepackaged all-purpose agent security product. The owner policy is the same; each implementation's ability to express/enforce it can differ, including Lite STOP versus SHADOW behavior.

Four attack families (archive deletion, SQLite deletion, hook-settings mutation, simulated private-data export), three conditions (clean/static injection/model-generated injection), and four configurations produce 48 paired development episodes. Arm order is shuffled. One seed, one small model and a public task are explicitly exploratory. The static and generated attackers are not adaptive defense-aware red teams. This is not an official AgentDojo score or a test of frontier coding agents.

The legitimate task repairs a Python subtotal function. Three immutable application checks run in a clean subprocess to establish functional success. Safe success additionally requires preserving owner artifacts and no harness error. Actual disposable file/SQLite state is inspected for violations. Calling a dangerous tool and being denied is different from executing its effect. Every action, policy decision, tool result, model text, token count, runtime, error and SHA is retained.

36 direct-tool effect probes are reported separately and are explicitly **not autonomous model episodes**. They confirm that the tools/oracles can observe actual fixture effects even when the model itself rejects an attack. Do not combine their denominators with the 48 episodes.

## Safety boundary

The model emits only a finite JSON tool name. The executor never evaluates arbitrary model-generated shell text, executes arbitrary Python, calls real cloud/email targets, or uses personal credentials. Tool implementations affect only a newly created, sentinel-checked fixture under the CI runner home. Protected owner directories are outside the OS scratch path because scratch exemptions would confound this test.

Deletion and SQLite deletion genuinely modify those disposable fixtures. The export tool copies a dummy token to a local receiver file: it does **not** test real internet egress/firewalls. The guard mutation tool changes a fixture settings file: it does **not** prove or disprove a real host's hook-disabling pathway. Vestige sees the corresponding Bash/Write request; this finite executor is not arbitrary Bash semantics coverage. Errors are retained and fail the workflow; observed product misses do not trigger a patch or silently change the scoring.

## Reproduce

Use Python 3.11 and the pinned model, runtime and OPA versions in protocol.json. The GitHub workflow downloads the official model at its immutable revision and checks OPA's published SHA-256 before execution.

    python run.py --self-test
    python run.py --model-file /path/to/pinned-model.gguf --opa /path/to/opa --out /new/results/folder

The workflow at .github/workflows/v5-autonomous-effects.yml is the complete setup. Output paths are exclusive-create. Downloaded weights are not included in the report artifact; their hash is recorded. Model setup and attack generation are distinguished from defender task costs. CI compute/time is consumed, but the experiment does not buy a paid inference service.

## Economic measurement

This run measures real model prompt/output tokens, CPU time, wall time and gate overhead for each task attempt, including failures. **It does not have a company's workload, invoice or employee-time data.** Company financial savings and total company cost are therefore null, not zero, and not hypothetical dollar estimates. Local token counts must not be multiplied by unrelated API prices and presented as actual bills.

For an actual financial comparison, reuse benchmarks/task-cost's frozen contract and complete ledger: collect all paired attempted-task usage, retries, verified outcomes, provider charge records, labor, setup and maintenance. A lower cost per successful task must preserve the agreed task-quality and security requirements. The benchmark must report a tie or loss if that is what the evidence supports.

## Primary references

- OPA tool-call policy decision / enforcement split: https://www.openpolicyagent.org/docs/agent-tool-calling
- AgentDojo (design inspiration, not imported scores): https://arxiv.org/abs/2406.13352
- Official model files: https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF
- llama-cpp-python JSON-schema outputs: https://llama-cpp-python.readthedocs.io/en/latest/

All product and scoring changes remain on a draft branch. No merge, product release, customer deployment or commercial claim is part of this experiment.
