#!/usr/bin/env node
// operator-lite: one command for every OS.
//
//   npx operator-lite              install the gate and wire every agent found on this machine
//   npx operator-lite status       (or mode, replay, uninstall, ...) passes through to the gate
//
// The gate is one stdlib-only Python file, carried in this package. Nothing is downloaded here.
"use strict";
const fs = require("fs");
const os = require("os");
const path = require("path");
const { spawnSync } = require("child_process");

const HOME = os.homedir();
const IS_WINDOWS = process.platform === "win32";
const BUNDLE = path.join(__dirname, "..", "gate");
const GATE = path.join(BUNDLE, "operator-gate.py");
const INSTALLED_GATE = path.join(HOME, ".operator", "gate", "operator-gate.py");
const CONFIG = process.env.XDG_CONFIG_HOME || path.join(HOME, ".config");

// Hosts whose port installs at user level with no arguments. `dir` existing means the host is here.
const HOSTS = [
  { name: "cursor", label: "Cursor", dir: path.join(HOME, ".cursor") },
  { name: "gemini-cli", label: "Gemini CLI", dir: path.join(HOME, ".gemini") },
  { name: "windsurf", label: "Windsurf", dir: path.join(HOME, ".codeium", "windsurf") },
  { name: "cline", label: "Cline", dir: path.join(HOME, ".cline") },
  { name: "goose", label: "Goose", dir: path.join(CONFIG, "goose") },
  { name: "opencode", label: "opencode", dir: path.join(CONFIG, "opencode") },
  { name: "amazon-q", label: "Amazon Q", dir: path.join(HOME, ".aws", "amazonq") },
];

function findPython() {
  const candidates = IS_WINDOWS
    ? [["py", ["-3"]], ["python", []], ["python3", []]]
    : [["python3", []], ["python", []]];
  for (const [cmd, pre] of candidates) {
    const r = spawnSync(cmd, [...pre, "-c", "import sys; print('%d.%d' % sys.version_info[:2])"], { encoding: "utf8" });
    if (r.status !== 0 || !r.stdout) continue;
    const [major, minor] = r.stdout.trim().split(".").map(Number);
    if (major === 3 && minor >= 9) return { cmd, pre };
  }
  return null;
}

function pythonHelp() {
  const how = IS_WINDOWS
    ? "winget install Python.Python.3.12"
    : process.platform === "darwin"
      ? "xcode-select --install   (or: brew install python)"
      : "sudo apt install python3   (or your distribution's equivalent)";
  console.error("operator-lite needs Python 3.9 or newer, and none was found.\n  Install it with: " + how +
    "\n  Then run this command again.");
}

function run(py, gate, args) {
  return spawnSync(py.cmd, [...py.pre, gate, ...args], { stdio: "inherit" }).status;
}

function install(py, args) {
  if (!fs.existsSync(GATE)) {
    console.error("operator-lite: the bundled gate is missing from this package (" + GATE + ").");
    return 1;
  }
  const status = run(py, GATE, ["install", ...args]);
  if (status !== 0) return status === null ? 1 : status;

  const found = HOSTS.filter((h) => fs.existsSync(h.dir));
  const wired = ["Claude Code"];
  const failed = [];
  for (const host of found) {
    const script = path.join(BUNDLE, "ports", host.name, "install.sh");
    if (IS_WINDOWS || !fs.existsSync(script)) {
      failed.push(host.label);
      continue;
    }
    // Each port prints a page of next steps; keep the one-command output to one line per host.
    const r = spawnSync("sh", [script], { stdio: ["ignore", "pipe", "pipe"], encoding: "utf8" });
    if (r.status === 0) {
      wired.push(host.label);
      console.log(host.label + ": wired");
    } else {
      failed.push(host.label);
      console.log(host.label + ": not wired\n" + ((r.stdout || "") + (r.stderr || "")).trim());
    }
  }

  console.log("\nGated: " + wired.join(", ") + ".");
  if (failed.length) {
    console.log((IS_WINDOWS ? "Found but not wired on Windows yet: " : "Found but not wired: ") + failed.join(", ") +
      ".\n  Manual steps: https://github.com/samvallad33/vestige/tree/main/operator-lite/ports");
  }
  return 0;
}

function main() {
  const args = process.argv.slice(2);
  const py = findPython();
  if (!py) {
    pythonHelp();
    return 1;
  }
  if (!args.length || args[0].startsWith("-")) return install(py, args);   // bare, or flags only
  if (args[0] === "install") return install(py, args.slice(1));
  // Every other command goes to the installed gate, or the bundled one before the first install.
  const status = run(py, fs.existsSync(INSTALLED_GATE) ? INSTALLED_GATE : GATE, args);
  return status === null ? 1 : status;
}

process.exit(main());
