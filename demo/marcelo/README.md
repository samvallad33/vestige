# Marcelo demo

Vestige is a cognitive, deterministic memory-transaction security OS for AI agents.

This script checks out pre-release commit `b4bcd52b81` from PR #445, builds the default-feature binaries, and runs the uv #10186 regression from the repository tests. It records that repository history, saves the failure record the test saves, walks backward from it, replays the signed log, and runs `strata-verify`.

Linux or macOS, with `git`, `cargo`, and `python3`:

```sh
bash demo/marcelo/run-demo.sh
```

`DEMO_PAUSE=1` waits about 2 seconds between steps:

```sh
DEMO_PAUSE=1 bash demo/marcelo/run-demo.sh
```

The script clones into a temporary directory and uses a new empty data directory. After the clone and the build, it stays on local files. A clean run exits 0 with the recorded cause at rank 1 and a MATCH line for the two digests.
