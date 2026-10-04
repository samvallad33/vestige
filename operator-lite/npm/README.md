# operator-lite

A deterministic pre-tool gate for coding agents. One command, the same on macOS, Linux and Windows:

    npx operator-lite

It installs the gate to `~/.operator/gate`, wires Claude Code, then wires every other agent it
finds on the machine (Cursor, Gemini CLI, Windsurf, Cline, Goose, opencode, Amazon Q). It starts in
shadow mode: every verdict is recorded and nothing is blocked until you run

    npx operator-lite mode enforce

Run it in your own terminal. The installer refuses to run inside an agent session.

Requires Python 3.9 or newer (the gate is one stdlib-only Python file, carried in this package).
On Windows, Claude Code is wired; the other hosts are listed with a link to their manual steps.

Other commands pass through to the gate: `status`, `replay`, `mode`, `approve`, `verify`, `uninstall`.

Full documentation: https://github.com/samvallad33/vestige/tree/main/operator-lite
