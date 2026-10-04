# Setting Up CLAUDE.md for Vestige

> Written for Vestige 4.x.

> Make Claude use Vestige without being asked.

Vestige finds a record by an exact handle: its full id, a unique id prefix of 8 or more characters, or an exact tag. It does not search by wording. So the one habit that matters is this: **tag every record you save, and recall by that tag.** The templates below are built around it.

---

## Quick Setup

Add this to your global `~/.claude/CLAUDE.md` or a project-level `CLAUDE.md`:

```markdown
## Vestige

At the start of every conversation:
1. Call `session_start` with `include_intentions: true` and `include_status: true`.
   In a repository, also pass `context: {codebase: "<repo>", repoPath: "<absolute path>"}`.
   Do not pass `queries`. Vestige ignores them and says so.
2. Call `recall` with `handle: "<narrow topic tag>"` for the task at hand.

Save decisions, corrections and verified facts with `smart_ingest` as they happen.
Always give each save a project tag and one narrow topic tag.
```

---

## Full Template (Recommended)

```markdown
# Vestige

Vestige is the log of what we decided and what happened. Use it automatically.

---

## 1. SESSION START

1. `session_start` with `include_intentions: true`, `include_status: true`, and
   in a repo `context: {codebase, repoPath}`. Never pass `queries`.
2. `recall` with `handle: "<narrow topic tag>"`, for example `handle: "payments"`.
   A tag recall returns every record under that tag, from every scope, with full
   content. Never recall a broad tag such as `decision` or `fact`.
3. Deal with any triggered intentions that `session_start` lists. Ask me before
   you cancel one.

A miss returns `error: handle_required`. That means no such tag or id exists.

---

## 2. SAVE WHEN IT HAPPENS

Save with `smart_ingest`. Give `content`, `node_type`, `tags` and `source`.

| Save this | node_type |
|---|---|
| A decision I make, with why and what was rejected | `decision` |
| A correction from me, or a fact you had wrong | `correction` |
| A verified fact, with the command or source that verified it | `fact` |
| A failure and its evidence-linked cause | `event` |
| A standing rule or preference of mine | `decision` |
| A finished milestone, with proof | `event` |

- Content must stand alone: the absolute date, what happened, why, the evidence
  (commands, ids, URLs, commit hashes) and how to apply it next time.
- Tags are the only index. One project tag, one or two narrow topic tags, spelled
  exactly the same every time. Tags are case-sensitive. No date tags.
- Recall the topic tag before you save, so the same fact is not stored twice.
- To declare a cause, pass `links` to existing records in the same scope:
  `derived_from`, `evidence_of` or `closes`. Those edges are what `causal_walk`
  follows.
- Never save secrets, raw logs, speculation or progress notes.

For a code convention or design decision, use `codebase` instead, and always pass
`repoPath`:

- `codebase` with `action: "remember_decision"`: decision, rationale, files
- `codebase` with `action: "remember_pattern"`: name, description, files

---

## 3. TRIGGERS

| I say or do | You call |
|---|---|
| "Remember this", or I state something important | Recall the topic tag, then `smart_ingest` |
| A goal, deadline or "remind me" | `intention` with `action: "set"` |
| "That was valuable" | `memory` with `action: "promote"` and the record's id |
| "That's wrong" | `memory` with `action: "demote"`, then `memory` with `action: "edit"` to admit the corrected version. The id changes |
| A test, CI run or bug fails | Save the failure as an `event`, then `causal_walk` from that record, then `forgotten_lesson` |

---

## 4. HYGIENE

- Promote when I confirm a record helped. Demote when it was wrong. A demote does
  not delete: the record fades faster.
- `suppress` hides a record from every read and cannot be undone on Strata. Ask me first.
- Check health with `memory_status` and `view: "health"`.
```

---

<a id="identity--autopoiesis-advanced"></a>
## Standing identity notes (Advanced)

If you want Claude to keep standing preferences, such as your style or the libraries you use, store them as records under one fixed tag and recall that tag at session start. There is no search for "who I am"; the tag is the handle.

```markdown
### Preferences (every session start)
1. `recall` with `handle: "preferences"`.
2. `intention` with `action: "check"` for pending commitments.

### Saving a preference
smart_ingest(
  content="2026-10-04. Prefer small focused PRs and explicit verification receipts. Why: reviews stay short.",
  node_type="decision",
  tags=["preferences", "pr-style"]
)

### Changing one
`memory` with `action: "edit"` and the record's id. The edit admits a successor and
retires the old record, so use the new id from then on.
```

---

## Project-Specific Setup

Add to your project's `CLAUDE.md`:

```markdown
## Vestige for this project

### On session start
- `codebase` with `action: "get_context"`, `codebase: "[project-name]"`, `repoPath: "[absolute path]"`
- `recall` with `handle: "[project-name]"`

### When making decisions
- `codebase` with `action: "remember_decision"` for every architectural choice
- Include the decision, the rationale, the alternatives considered and the affected files
- Always pass `repoPath`. Without it the anchor is stored unverifiable

### Patterns
- `codebase` with `action: "remember_pattern"` for recurring code patterns
- Include the pattern name, when to use it and example files

### After a refactor
- `codebase` with `action: "verify"` and `repoPath`. Re-anchor what moved with `action: "reanchor"`
```

`get_context` marks each code record as current or stale by comparing its anchors with the files in your checkout. It compares exact source spans, not meaning. See [CODE-CONTEXT-EVIDENCE.md](CODE-CONTEXT-EVIDENCE.md).

---

## Tips

- Run `memory_status` with `view: "tools"` to list the tools. Add `tool: "<name>"` for one tool's full schema.
- The `project` tool can write the durable subset of your records into a fenced region of `CLAUDE.md`. Preview first. See [PROJECTION.md](PROJECTION.md).
- Rules in a `CLAUDE.md` are guidance. If you need a command stopped before it runs, that is a hook, such as [Operator Lite](../operator-lite/README.md).
