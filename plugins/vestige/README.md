# Vestige Claude Code plugin

One install loads the Vestige MCP server, the Reflex hook fragments from `hooks/settings.*.fragment.json`, and the Operator Lite pre-tool gate.

```text
/plugin install vestige@vestige
```

The same line from a shell:

```bash
claude plugin install vestige@vestige
```

Add the marketplace once if it is not already known:

```bash
claude plugin marketplace add samvallad33/vestige
```

## Team rollout

Commit this in the repository's `.claude/settings.json`. After a collaborator trusts the folder, Claude Code registers the marketplace and prompts them to install the plugin:

```json
{
  "extraKnownMarketplaces": {
    "vestige": {
      "source": {
        "source": "github",
        "repo": "samvallad33/vestige"
      }
    }
  },
  "enabledPlugins": {
    "vestige@vestige": true
  }
}
```

This repository commits that file. `enabledPlugins` turns the plugin on for the project. A `github` marketplace still needs one `claude plugin install vestige@vestige --scope project` per clone. Hooks register on the next session, or when the session runs `/reload-plugins`.

## What the plugin starts

- MCP server `vestige` (`plugin:vestige:vestige` in `/mcp`). `bin/vestige-mcp.js` runs `vestige-mcp` when it is already on `PATH`. Otherwise it runs `npx -y vestige-mcp-server@<plugin version>`. npm `vestige-mcp-server` latest is still 5.0.1; the plugin version is 6.2.2, so the npx fallback works after that release is published. Until then, put the 6.2.2 `vestige-mcp` binary on `PATH`.
- Hook fragments, copied byte-for-byte from `hooks/`: rules (`SessionStart`), stop-memory, preflight, and Sanhedrin. The empty `settings.fragment.json` is bundled and not loaded.
- Operator Lite on `PreToolUse`. The hook runs `~/.operator/gate/operator-gate.py hook --source claude` when that file exists, and exits 0 when it does not, so a machine that has not run `npx @vestige/init --guard` is not blocked. Install the gate with `npx @vestige/init --guard` from an owner terminal.
