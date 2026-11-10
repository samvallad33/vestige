# Operator gate

One deterministic pre-tool gate. `npx @vestige/init --guard` installs it when
`~/.operator/gate/operator-gate.py` is not already there.

## What a fresh install writes

| File | What it is |
|---|---|
| `~/.operator/gate/operator-gate.py` | Operator 6.2.1. 102 STOP rules, OP-000 through OP-114. OP-086–OP-097 and OP-100 are reserved and are not rules. |
| `~/.operator/gate/lite_core.py` | Operator Lite from `operator-lite/operator-gate.py`, byte for byte. The gate loads it only when its sha256 is `f42156e9c9c8c6e5f4f4cdc6c2a698f49925338306e741b6645cdf417fbf50a1`. |

An existing gate file is not replaced. Delete it yourself and run `--guard` again to install this one.

The gate classifies tool calls. It does not call a model and it does not open a network connection. A STOP exits 2 and writes the reason on stderr. Receipts are hash-chained digests (`integrity = "reference_digest_not_signature"`), not signatures.

`board`, `letter`, `almost-forgot`, `onboard` and `briefing` load companion modules from the gate's directory when those files are present. The 102 rules do not need them.

## Check the rule count

From a clone:

```bash
python3 operator/operator-gate.py status --rules
python3 operator/test_rules.py
```

`test_rules.py` loads the 102 numbered rules and fires each one on a representative command. The commands are classified and not executed.
