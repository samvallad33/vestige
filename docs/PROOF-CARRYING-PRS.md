# Proof-carrying pull requests

When an agent says the tests pass, Vestige can hand the pull request the proof. Every allowlisted test
run becomes a signed witness bound to the exact git tree it ran on. One call bundles those witnesses for
a commit, and a GitHub Action checks the bundle offline before the pull request can merge.

No claim in a bundle comes from what anyone wrote about the run. Each claim is derived from a signed
witness, and a command with no witness shows as `unwitnessed`.

## How it works

1. **Witness.** Reflex watches every shell call through the hooks `@vestige/init` installs. When the
   whole command equals an entry on your allowlist, it records a witness: the command, its exit code,
   the SHA-256 of the output the host reported, and the git tree just before and just after the run.
   The witness is signed by your store's actor key and saved as a memory tagged `tree-<tree>`.
2. **Prove.** `receipt(action='prove')` takes the witnesses bound to one commit's tree, adds the
   Failure Pairing `closes` edges and the Operator stops from the same sessions, and writes them as an
   in-toto Statement v1 inside a DSSE envelope.
3. **Verify.** `vestige-verify-proof`, `vestige verify-proof` and the GitHub Action check the bundle
   against keys you pin. They read the bundle and local git, and they do not open a Vestige store.

## Set it up

### 1. Allowlist your test commands

Put the commands your project counts as its tests in `.vestige/witness.toml`:

```toml
[witness]
commands = ["cargo test --workspace", "pnpm test"]

# Optional: a junit report written by the run, used to count the tests.
[witness.junit]
"cargo nextest run --profile ci" = "target/nextest/ci/junit.xml"
```

The same `[witness]` table in the `vestige.toml` of your data folder applies to every repository.

A command is witnessed when the whole command the agent ran equals an entry, after surrounding spaces
are trimmed. `cargo test --workspace | tail -5` is a different command, because its exit code is the
exit code of `tail`. Background runs and interrupted runs are skipped. Name the exact command in your
agent instructions, such as `CLAUDE.md`, so the agent runs it as written: in a live check an agent
asked to run the tests chose `npm --prefix <path> test`, which is not `npm test`.

### 2. Pin your key

Bundles are signed with the actor key of your store, `actor.key` in your Vestige data folder. Its
public half is a `did:key`, and every `prove` result names it as `signer`. Put it on your main branch:

```text
# .vestige/trusted-keys
did:key:z6Mk...
```

### 3. Let the agent run the tests

Nothing else is needed. After an allowlisted run the agent sees one line, here from a live Claude Code
session:

```text
Reflex: witnessed `npm test` green at tree 61151e05a092, signed receipt mem-0000000000000031.
```

### 4. Prove the commit

```json
{"name": "receipt", "arguments": {"action": "prove", "repoPath": "/path/to/repo", "commit": "HEAD"}}
```

This writes `.vestige/proof/<tree>.intoto.jsonl`. Commit it with the pull request. The folder
`.vestige/proof` is left out of every tree hash, so committing the bundle keeps the tree it proves.

To keep bundles out of the tree, pass `"out": "notes"` and push the note with
`git push origin refs/notes/vestige-proof`. `"out": "both"` writes both.

### 5. Check pull requests

```yaml
name: Proof
on: pull_request

jobs:
  proof:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with:
          ref: ${{ github.event.pull_request.head.sha }}
          fetch-depth: 0
      - uses: samvallad33/vestige/actions/proof-carrying-pr@main
```

The Action builds the verifier from its own checkout. It reads `.vestige/trusted-keys` and
`.vestige/witness.toml` from the base branch, so a pull request cannot change the keys or the commands
it is checked against. Every command in that `witness.toml` must be green. The result goes to the job
summary as a table. Pin `@main` to a commit SHA for stable builds.

| Input | What it does |
|---|---|
| `trusted-keys` | More `did:key` values, one per line |
| `require` | More commands that must be green, one per line |
| `strict` | `true` makes every allowlisted command in the bundle required |
| `commit` | The commit to check, by default the pull request head |
| `base` | The branch keys and commands are read from, by default the pull request base |

## Statuses

| Status | When |
|---|---|
| `green` | The newest witness bound to the tree exited 0, and the runner did not report an empty run |
| `red` | The newest bound witness exited non-zero, or failed without an exit code |
| `zero_tests` | It exited 0, and the runner's JSON or junit said no test ran |
| `unwitnessed` | The command is on the allowlist and has no witness bound to the tree |

A witness is bound to a tree when the tree was the same before and after the run. A run that changed
tracked files counts for no tree, so build outputs belong in `.gitignore`.

## Zero tests

A run that collects zero tests often exits 0. Reflex reads the count from the runner's own structured
output, and plain text gives no count:

| Runner output | Read |
|---|---|
| Jest or Vitest `--json` | `numTotalTests`, `numPassedTests`, `numFailedTests` |
| Mocha `--reporter json` | `stats.tests`, `stats.passes`, `stats.failures` |
| pytest-json-report | `summary.collected` |
| libtest JSON, cargo or nextest | the suites' `test_count`, `passed`, `failed` |
| `go test -json` | each distinct test that ran |
| junit XML from `[witness.junit]` | `tests`, `failures`, `errors`, `skipped`, when the file was written during the run |

## What the verifier checks

1. The file is one DSSE envelope in its canonical form, signed once by a pinned key whose `did:key` is
   the envelope's `keyid`.
2. It holds an in-toto Statement v1 with the predicate type
   `https://vestige.dev/attestations/proof-carrying-pr/v1` and one subject, `{"gitTree": "<tree>"}`.
3. That tree is the commit's tree without `.vestige/proof`.
4. Every witness envelope inside opens against a pinned key and is bound to the tree.
5. The claims equal what those witnesses give. A signer cannot add a green claim without a witness.
6. At least one claim is green, none is red or `zero_tests`, and every required command is green.

Changing any single byte of a bundle fails the check. The test suite flips every byte of a bundle
three ways and checks that each one fails.

```text
$ vestige-verify-proof --repo . --trust did:key:z6Mkqp3uw9RCWJrU8xiGgHPUX7aQmrEVyGE9iWmPRZpDTXXX
vestige-verify-proof: PASS  tree 61151e05a092  commit f8a9cf060cb2  signer did:key:z6Mkqp3uw9RCWJrU8xiGgHPUX7aQmrEVyGE9iWmPRZpDTXXX
  green        npm test  witness mem-0000000000000031  2 tests
  4 witnesses, 1 Failure Pairing closes edges, 0 Operator stops, 3 sessions
```

That bundle came from three live sessions on one tree: a red run, a run the agent fixed, and a green
run. All four witnesses stay in the bundle, the newest decides the claim, and the fix is the Failure
Pairing edge.

Exit codes are 0 for a pass, 1 for a bundle that fails or is missing, and 2 for arguments that cannot
be used. `--format json` and `--format markdown` print the same report as data or as a table.

## What a bundle shows

The signature shows which exact bytes the pinned key signed. Each claim follows from a witness that key
signed when the command finished, on the tree it names. A bundle does not show that the tests are
meaningful, and it cannot rule out a second bundle signed by the same key.

The key sits in your Vestige data folder. Keep it as private as an SSH key, and pin only the keys of
machines you trust. Each bundle carries this boundary in its `claimBoundary` field.

## The bundle

`.vestige/proof/<tree>.intoto.jsonl` is one JSON line: a DSSE envelope with payload type
`application/vnd.in-toto+json`. Its statement's predicate holds:

| Field | What it holds |
|---|---|
| `tree`, `commit` | The tree proved and the commit `prove` was asked about |
| `claims` | One claim per command: `status`, the `witness` memory, `exit`, `tests`, `runs` |
| `allowlist` | The allowlist in force at that commit |
| `witnesses` | Every bound witness: its memory, its Strata payload digest, its signed envelope |
| `pairings` | Failure Pairing `closes` edges from the same sessions: failure, fix, signature |
| `stops` | Operator stops from the same sessions, with the resolution that closed each one |
| `sessions`, `producer`, `generatedAt`, `claimBoundary` | Where the bundle came from and what it shows |

Each witness payload, type `application/vnd.vestige.witness.v1+json`, holds the command, `exit`,
`outputSha256` and `outputBytes`, `tree` and `treeBefore`, `head`, `dir`, `tests`, the host, session,
call and agent, the start and end times, and the producer. Anyone with the same output can recompute its
digest: it is the SHA-256 of the canonical JSON `{"stderr":…,"stdout":…}` the host reported, or
`{"error":…}` for a failed call.
