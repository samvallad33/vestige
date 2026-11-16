# Claude Code: rules that survive the compact

> Written for Vestige 4.2.x. Opt-in. Nothing is enabled by default.

> Your pinned decisions come back after every `/compact`, `/clear` and resume, by exact tag.

Compaction rewrites a Claude Code conversation into a summary. Whatever the summary
drops is gone for the rest of the session, including the rules you told the agent to
keep. `vestige hook session-start` is a plain command hook for Claude Code's
`SessionStart` event. Claude Code runs it when a session starts, resumes, is cleared,
is forked, and right after a compaction (`source: "compact"`). It prints the default
rulebook and your pinned rules back into the context, each rule with its memory id.

---

## What it does

1. Reads the hook payload Claude Code sends on stdin: `session_id`, `cwd`, `source`.
2. Works out the **exact tags**:
   - `pinned`;
   - each line of an optional `.vestige-rules` file in `cwd` (or the repository root):
     one exact tag per line, `#` for comments;
   - only with `--repo-tag`: the git repository name of `cwd` (the `origin` remote's
     name, else the main checkout's folder name, else the folder name). A capitalised
     name is also tried in lowercase, as a second exact tag. It is off by default: when
     every memory of a project carries the project tag, that tag holds far more than
     rules, and its newest entries would crowd out the ones you pinned.
3. Resolves each tag with `recall` `handle`, exactly as the MCP tool does, through the
   Vestige server that holds the store. When no server is running it reads the log
   directly.
4. Prints the hook JSON:

   ```json
   {"hookSpecificOutput":{"hookEventName":"SessionStart","additionalContext":"..."}}
   ```

   The context lists the `decision` and `correction` memories under those tags, newest
   first, at most 8 entries and about 4 KB. Each line starts with the memory id, so the
   agent can open the full memory with `recall(handle='<id>')` and its receipt on the
   hash-chained log with `receipt(action='get', receipt_id='<id>')`.

Real output for a `compact` payload from inside a `tidetable` checkout, from a test store
with one rule tagged `pinned` and one tagged `tidetable`:

```
Vestige rules for this session (source: compact). These are the decision and correction memories recorded under the exact tags [pinned], newest first. They were resolved by exact tag through recall(handle=...), not by search or similarity. Each line starts with its memory id; recall(handle='<id>') returns the full memory and receipt(action='get', receipt_id='<id>') its receipt on the hash-chained log. The conversation was just compacted: these rules are restated from the log, whatever the summary kept.
- mem-0000000000000001 [decision, tag pinned, 2026-10-16] Tide heights in tidetable are always stored and printed in metres, never feet. Rule codeword: HALIBUT-7.
```

With `--repo-tag` (or `tidetable` listed in `.vestige-rules`), the same run also lists the
`tidetable` rule:

```
Vestige rules for this session (source: compact). These are the decision and correction memories recorded under the exact tags [tidetable, pinned], newest first. ...
- mem-0000000000000005 [correction, tag tidetable, 2026-10-16] Correction: tidetable rounds every tide height to two decimals before printing. Rule codeword: MARLIN-3.
- mem-0000000000000001 [decision, tag pinned, 2026-10-16] Tide heights in tidetable are always stored and printed in metres, never feet. Rule codeword: HALIBUT-7.
```

The date is the UTC day the memory was written.

## What it does not do

- It never searches. A memory is listed only when it carries one of those tags byte for
  byte (tags are case-sensitive). Words in a memory's text, a near-miss tag, or a
  sentence in `.vestige-rules` match nothing.
- It never writes: no memory, no edge, no receipt. A test checks the log's bytes are
  unchanged after a run.
- It lists only `decision` and `correction` memories. Facts, notes and events under the
  same tags stay out.
- It does not record which handles a session held before compaction or diff them
  against the summary. That is the fuller design in issues #396 and #397 (PreCompact
  checkpoint, PostCompact audit, re-injecting only the dropped handles). This hook
  re-injects the whole pinned set every time.
- It fails open. No store, no matching rule, a malformed payload, an error, or its 5 s
  internal limit: it prints nothing and exits 0, so it can never block a session.

---

## Install

The `vestige` binary must be on the `PATH` Claude Code runs hooks with. Merge
[`hooks/settings.rules.fragment.json`](../../hooks/settings.rules.fragment.json) into
`~/.claude/settings.json` (all projects) or `.claude/settings.json` (one project):

```json
{
  "hooks": {
    "SessionStart": [
      {
        "matcher": "startup|resume|clear|compact|fork",
        "hooks": [
          { "type": "command", "command": "vestige hook session-start", "timeout": 10 }
        ]
      }
    ]
  }
}
```

If you already have `SessionStart` hooks, add this group to the existing array; do not
replace it. To use a store other than the default, put `--data-dir <dir>` after
`session-start`. On Windows the same command works; no shell or Python wrapper is
involved.

Check it by hand from a project directory:

```bash
printf '{"hook_event_name":"SessionStart","source":"compact","cwd":"%s"}' "$PWD" \
  | vestige hook session-start
```

---

## Pin a rule

Save the rule as a `decision` or `correction`, tagged `pinned` (every project) or with a
narrow tag you list in `.vestige-rules` (this project only). Through the MCP tools, in any
agent:

```
smart_ingest(
  content="2026-10-15: tide heights are stored and printed in metres, never feet. Why: the harbour API is metric.",
  node_type="decision",
  tags=["tidetable", "pinned"],
  source="team decision"
)
```

Or from a terminal, when no Vestige server holds the store:

```bash
vestige ingest "Never push to main without the fixture tests passing." \
  --node-type correction --tags pinned
```

For project-specific tags, list them in `.vestige-rules` at the repository root:

```
# exact tags, one per line
codebase:tidetable
release-policy
```

`codebase:<repo>` is the tag `codebase(action='remember_decision')` adds, so code
decisions saved that way come back too. To bring back everything under the repository's
own tag, list that tag here or pass `--repo-tag`.

## Options

| Option | Default | What it does |
|---|---|---|
| `--max-items <n>` | 8 | Most memories listed |
| `--max-bytes <n>` | 4096 | Most bytes of context (clamped to 512..9000; Claude Code caps a hook's context at 10,000 characters) |
| `--timeout-ms <ms>` | 5000 | Give up, print nothing and exit 0 after this long |
| `--repo-tag` | off | Also read the git repository name of `cwd` (and its lowercase spelling) as an exact tag |
| `--data-dir <dir>` | the platform data directory | Which store to read |

Option values go after `session-start`.
