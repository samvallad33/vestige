# operator-lite

A deterministic pre-tool gate for coding agents. One command, the same on macOS, Linux and Windows:

    npx operator-lite

It installs the gate to `~/.operator/gate`, wires Claude Code, then wires every other agent it
finds on the machine (Cursor, Gemini CLI, Windsurf, Cline, Goose, opencode, Amazon Q). It starts in
shadow mode: every verdict is recorded and nothing is blocked until you run

    npx operator-lite mode enforce

Run it in your own terminal. The installer refuses to run in a shell that Claude Code's agent
started (it reads `CLAUDECODE` and `CLAUDE_CODE_ENTRYPOINT`), or in any shell that sets
`OPERATOR_AGENT_SESSION`. Other agents' shells are not recognized, so do not hand it to an agent.

It never replaces a gate that outranks it. When `~/.operator/gate/operator-gate.py` is the paid
Operator gate, or a newer version, it prints both versions and the path, changes nothing, and exits 4.
`npx operator-lite --force` replaces it anyway: it warns, keeps the replaced gate in
`~/.operator/install-backups/`, and prints the command that puts it back.

Requires Python 3.9 or newer (the gate is one stdlib-only Python file, carried in this package).
On Windows, Claude Code is wired; the other hosts are listed with a link to their manual steps.

Other commands pass through to the gate: `status`, `replay`, `mode`, `approve`, `verify`, `uninstall`.
They run the installed gate when there is one (which may be the paid Operator), else the gate in this
package, and each first prints which gate ran, with its version and path, on stderr.

Full documentation: https://github.com/samvallad33/vestige/tree/main/operator-lite
