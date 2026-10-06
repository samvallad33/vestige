# Marcelo demo

Vestige is a cognitive, deterministic memory-transaction security OS for AI agents.

These scripts check out pre-release commit `b4bcd52b81` from PR #445 and run the uv #10186 regression from the repository tests. History is read at the parent of `b52d489`, so that revert commit stays outside the record. The observed revision `351d602` and the recorded cause `d2f58d9` stay inside it.

Linux or macOS, with `git`, `cargo`, and `python3`. Run setup once:

```sh
bash demo/marcelo/setup.sh
```

The last line is `ready`. The default folder is `$HOME/vestige-demo`. Set `DEMO_HOME` to use another one. Leave `DEMO_HISTORY` unset for the test page of 20 commits. `DEMO_HISTORY=300` ingests 300 commits ending at the same parent; run setup with that variable set so the checkout holds them.

Then, in this order:

```sh
DEMO_PAUSE=1 bash demo/marcelo/vestige-side.sh
bash demo/marcelo/rag-side.sh
```

`DEMO_PAUSE=1` waits about 2 seconds between steps on the Vestige side. That side uses a fresh empty data directory, records the failure the test records, walks backward from it, replays the signed log, and runs `strata-verify`. It exits non-zero when the recorded cause is not rank 1 or the digests differ.

`rag-side.sh` is a separate baseline in `demo/marcelo/rag_baseline/`. It is outside the Vestige crates. It reads the same commits and the same failure text the Vestige side just wrote. After setup, both commands stay on local files.

`run-demo.sh` is setup plus the Vestige side:

```sh
bash demo/marcelo/run-demo.sh
```

The recording notes are in `demo/marcelo/RECORDING.md`.
