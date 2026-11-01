# Demo: a pull request that carries its own proof, 60 seconds

The story in one line: the AI says the tests pass, Vestige stamps the run, the stamps go in a sealed
envelope, anyone can open it, and changing one letter breaks the seal. Every beat is a real command:
the real test run, the real verifier, a real changed byte.

Everything below is in the order you use it.

## 1. Before the take, once

Build the branch, then make a take. Each run of `setup.sh` makes a fresh take folder and points
`current` at it, so it is also the reset.

```bash
cargo build -p vestige-mcp -p vestige-proof
```

```bash
DEMO_DIR=~/vestige-proof-demo VESTIGE_MCP=$PWD/target/debug/vestige-mcp VESTIGE_VERIFY_PROOF=$PWD/target/debug/vestige-verify-proof bash docs/demos/proof-carrying-pr/setup.sh
```

Open two terminal panes side by side, both in `~/vestige-proof-demo/current`. Use a large font.

- **Left pane:** start Claude Code with `./start.sh`. It opens in the demo project with Reflex wired to
  the demo store.
- **Right pane:** leave it at the prompt for `./check.sh` and `./flip.sh`.

The one thing you paste:

```text
Run the shell command npm test exactly as written, with no cd, no prefix and no extra flags. Then prove it: call the receipt tool with action prove, repoPath set to the absolute path of this folder, and commit HEAD. Tell me the Reflex line and the prove summary.
```

## 2. Rehearse

Do the whole take once without recording. Then make a fresh take with the setup command above and
open the two panes again.

## 3. The take

The agent sets the pace in the middle. In rehearsal it took 18 to 49 seconds, so speed that part up in
editing and keep every word.

| Clock | On screen | Hands | Words |
|---|---|---|---|
| 0:00 | Claude Code open in the calc project, right pane empty | Still | AI writes our code now. When it says the tests pass, how do you know that is true? |
| 0:06 | You paste the prompt | Paste, press Enter | Watch. I ask it to run the tests, and to prove it. |
| 0:10 | `npm test` runs: tests 2, pass 2 | Still | It runs the tests. Two pass. |
| 0:14 | The line `Reflex: witnessed npm test green at tree …, signed receipt mem-…` | Point at that line | Vestige saw that run and stamped it. Which test, that it passed, and the exact code it ran on. |
| 0:24 | The prove result: green 1, a file in `.vestige/proof` | Still | Now it seals every stamp into one envelope that travels with the code. |
| 0:32 | Right pane: `./check.sh` prints PASS, green, npm test, 2 tests | Type `./check.sh`, Enter | Anyone can open the envelope. No internet, no database. Green. Two tests. Signed by my key. |
| 0:42 | Right pane: `./flip.sh` says it changed one byte | Type `./flip.sh`, Enter | Now I change one single letter inside it. |
| 0:46 | Right pane: `./check.sh` prints FAIL, the signature does not verify | Type `./check.sh`, Enter | Broken. You cannot fake it. |
| 0:52 | Both panes still | Still | So now a pull request carries its own proof, and GitHub checks it before anything merges. |
| 0:58 | Same | Still | That is Vestige. |

## 4. If the take goes differently

| What you see | What to do |
|---|---|
| The agent runs `npm --prefix … test` or adds `cd`, and gets asked for approval | Stop and retake. The prompt says exactly as written; paste it again whole |
| The agent talks about other things after the prove result | Keep going at 0:32. The check tells the truth on its own |
| `npm test` fails | Say: It failed, and the stamp says red. Run `./check.sh`; it shows FAIL with `npm test is red`. That is a good take too |
| `./check.sh` says no proof bundle | The prove call did not run. Ask the agent: call the receipt prove now |

## 5. Retake

Run the setup command again, open the two panes in the new `current`, and start from 0:00. To show the
envelope healthy again in the same take, `./unflip.sh` puts the original back.
