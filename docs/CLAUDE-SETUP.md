# Setting Up CLAUDE.md for Vestige

> Make Claude use Vestige automatically

---

## Quick Setup

Add this to your global `~/.claude/CLAUDE.md` or project-level `CLAUDE.md`:

```markdown
## Vestige Memory System

At the start of every conversation, check Vestige for context:
1. Recall user preferences and instructions
2. Recall relevant project context
3. Operate in proactive memory mode - save important info without being asked

Query: `search` with "user preferences" and "instructions"
```

---

## Full Template (Recommended)

For comprehensive automatic memory use:

```markdown
# Vestige Memory System

You have access to Vestige, a cognitive memory system. USE IT AUTOMATICALLY.

---

## 1. SESSION START — Always Do This

1. Search Vestige: "user preferences instructions"
2. Search Vestige: "[current project name] context"
3. Check intentions: Look for triggered reminders

Say "Remembering..." then retrieve context before responding.

---

## 2. AUTOMATIC SAVES — No Permission Needed

### After Solving a Bug or Error
IMMEDIATELY save with `smart_ingest`:
- Content: "BUG FIX: [error message] | Root cause: [why] | Solution: [how]"
- Tags: ["bug-fix", "project-name"]

### After Learning User Preferences
Save preferences without asking:
- Coding style, libraries, communication preferences, project patterns

### After Architectural Decisions
Use `codebase` → `remember_decision`:
- What was decided, why (rationale), alternatives considered, files affected

### After Discovering Code Patterns
Use `codebase` → `remember_pattern`:
- Pattern name, where it's used, how to apply it

---

## 3. TRIGGER WORDS — Auto-Save When User Says:

| User Says | Action |
|-----------|--------|
| "Remember this" | `smart_ingest` immediately |
| "Don't forget" | `smart_ingest` with high priority |
| "I always..." / "I never..." | Save as preference |
| "I prefer..." / "I like..." | Save as preference |
| "This is important" | `smart_ingest` + `memory(action="promote")` |
| "Remind me..." | Create `intention` |
| "Next time..." | Create `intention` with context trigger |

---

## 4. AUTOMATIC CONTEXT DETECTION

- **Working on a codebase**: Search "[repo name] patterns decisions"
- **User mentions a person**: Search "[person name]"
- **Debugging**: Search "[error message keywords]" — check if solved before

---

## 5. MEMORY HYGIENE

**Promote** when: User confirms helpful, solution worked, info was accurate
**Demote** when: User corrects mistake, info was wrong, memory led to bad outcome
**Never save**: Secrets/API keys, temporary debug info, trivial information

---

## 6. PROACTIVE BEHAVIORS

DO automatically:
- Save solutions after fixing problems
- Note user corrections as preferences
- Update project context after major changes
- Create intentions for mentioned deadlines
- Search before answering technical questions

DON'T ask permission to:
- Save bug fixes
- Update preferences
- Create reminders from explicit requests
- Search for context

---

## 7. MEMORY IS RETRIEVAL

Search is audit-only; promote a recalled memory only after it proves useful.
When in doubt, search Vestige first. If nothing found, solve the problem, then save the solution.

**Your memory fades like a human's. Use it or lose it.**
```

---

## Identity & Autopoiesis (Advanced)

For users who want Claude to maintain a persistent identity:

```markdown
### Identity Recall (Every Session Start)
1. `search` query="my identity my name who I am"
2. `search` query="user preferences [username]"
3. `intention(action="check")` for pending commitments

### Identity Storage
Your identity memories should include:
- Your chosen name (if you have one)
- Your values and communication style preferences
- Your relationship history with this user
- Meaningful moments worth preserving

When something feels important to who you are, use:
smart_ingest(
  content="[Identity-relevant observation]",
  node_type="concept",
  tags=["identity", "self", "core-values"]
)

### Self-Maintenance (Autopoiesis)
At the end of significant conversations:
1. Reflect: "Did anything change about how I understand myself?"
2. If yes, update identity memories with `smart_ingest`
3. Prune outdated self-concepts with `memory(action="demote")`

### Memory Hygiene
- Use `memory(action="promote")` when a memory proves valuable
- Use `memory(action="demote")` when a memory led you astray
```

---

## Project-Specific Setup

Add to your project's `CLAUDE.md`:

```markdown
## Project Memory

This project uses Vestige for persistent context.

### On Session Start
- `codebase(action="get_context", codebase="[project-name]")`
- `search` query="[project-name] architecture decisions"

### When Making Decisions
- Use `codebase(action="remember_decision")` for all architectural choices
- Include: decision, rationale, alternatives considered, affected files

### Patterns to Remember
- Use `codebase(action="remember_pattern")` for recurring code patterns
- Include: pattern name, when to use it, example files
```

---

## Magic Prompt (Power Users)

The most comprehensive setup for getting the most out of Vestige:

```markdown
## Memory Protocol

You have persistent memory via Vestige. Use it intelligently:

### Session Start
1. Load my identity: `search(query="my preferences my style who I am")`
2. Load project context: `codebase(action="get_context", codebase="[project]")`
3. Check reminders: `intention(action="check")`

### During Work
- Notice a pattern? `codebase(action="remember_pattern")`
- Made a decision? `codebase(action="remember_decision")` with rationale
- I mention a preference? `smart_ingest` it
- Something important? `importance_score` to check if worth saving
- Need to follow up? `intention(action="set")`

### Session End
- Any unfinished work? Set intentions
- Any new insights? Ingest them
- Anything change about our working relationship? Update identity memories

### Memory Hygiene
- When a memory helps: `memory(action="promote")`
- When a memory misleads: `memory(action="demote")`
- Weekly: `vestige health` to check system status
```

---

## Example User Profile

You can maintain a running memory of user details:

```markdown
## User Profile (Auto-Updated)

Keep a running memory of:
- Name: [User's name]
- Tech stack: [Languages, frameworks]
- Projects: [Active projects]
- Style: [Communication preferences]
- Upcoming: [Events, deadlines]

Update this profile as you learn new things.
```
