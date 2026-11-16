# Tweaks log

Gemini's pasted `docs/TWEAKS_FULL_LOOP.md` describes edits already present in the drop (fail-open `main`, `MissingProvenanceReceipt`, `SubagentPassport`). Those are not diffs against this repo. This file records only changes made while wiring the drop in.

## 1. `hooks/save-gate.py` instead of `hooks/operator-stop-memory.py`

- **File**: `hooks/operator-stop-memory.py` (drop path) → `hooks/save-gate.py`
- **Original text**: the drop is a new program saved at `hooks/operator-stop-memory.py`
- **New text**: the same bytes, at `hooks/save-gate.py`. `hooks/operator-stop-memory.py` is untouched.
- **Error that forced it**: the path is the Operator-stop relay. `packages/vestige-init/test/kernel.test.js` and `memory-setup.test.js` require it to be byte-identical to `packages/vestige-init/hooks/operator-stop-memory.py`, and `six-tool-memory-hooks.test.js` calls `ingest()` on it. Pointing `hooks/test_save_gate.py` at the existing file fails:

```
FAIL: test (__main__.T.test)
AssertionError: unexpectedly None
```

`PreToolUse` on `git commit -m 'fix'` after an edit produced no JSON (`rc 0`, empty stdout). The save-gate assertion `permissionDecision == deny` never ran.

## 2. `hooks/test_save_gate.py` hook path

- **File**: `hooks/test_save_gate.py`
- **Original text**: `HOOK = Path(__file__).with_name("operator-stop-memory.py")`
- **New text**: `HOOK = Path(__file__).with_name("save-gate.py")`
- **Error that forced it**: the assertion in tweak 1. `Path(__file__).with_name` loads the sibling of the test, which is the Operator relay.

## 3. `hooks/settings.save-gate.fragment.json` instead of `hooks/settings.stop-memory.fragment.json`

- **File**: `hooks/settings.stop-memory.fragment.json` (drop path) → `hooks/settings.save-gate.fragment.json`
- **Original text**: `"command": "python3 hooks/operator-stop-memory.py --event UserPromptSubmit"` (and the same command for `PreToolUse`, `PostToolUse`, `Stop`)
- **New text**: `"command": "python3 hooks/save-gate.py --event UserPromptSubmit"` (and the same command for the other three events). Matchers and `"timeout": 5` are unchanged.
- **Error that forced it**: `benchmarks/real_engine/integrity.py` `_registered_hook_catalog` reads `hooks/settings.stop-memory.fragment.json` by name. The file on `main` registers `python3 ~/.vestige/hooks/operator-stop-memory.py` at timeout 30, including `SessionStart`. Replacing it changes the catalog the benchmark checks and the fragment `npx @vestige/init` ships. Not edited.

## 4. Docstring path left as the drop wrote it

- **File**: `hooks/save-gate.py`
- **Original text**: `Vestige Save-Gate & 7-Field Strata Provenance Hook (hooks/operator-stop-memory.py)`
- **New text**: unchanged
- **Error that forced it**: none. The docstring is not a path the test or the fragment executes.
