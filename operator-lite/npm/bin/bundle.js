#!/usr/bin/env node
// prepack: copy the gate and the host ports into ./gate so the published package carries them.
// The layout under gate/ mirrors operator-lite/, so each port's install.sh finds its sibling gate.
"use strict";
const fs = require("fs");
const path = require("path");

const pkg = path.join(__dirname, "..");
const src = path.join(pkg, "..");
const dst = path.join(pkg, "gate");

fs.rmSync(dst, { recursive: true, force: true });
fs.mkdirSync(dst, { recursive: true });
fs.copyFileSync(path.join(src, "operator-gate.py"), path.join(dst, "operator-gate.py"));
fs.cpSync(path.join(src, "ports"), path.join(dst, "ports"), {
  recursive: true,
  filter: (p) => !/(^|[\\/])(__pycache__|node_modules)$/.test(p),
});

const gate = fs.readFileSync(path.join(dst, "operator-gate.py"), "utf8");
const version = (gate.match(/^VERSION = "([^"]+)"/m) || [])[1];
const want = require(path.join(pkg, "package.json")).version;
if (version !== want) {
  console.error(`bundle: package.json is ${want} but the gate is ${version}; they ship as one version.`);
  process.exit(1);
}
console.log(`bundled gate ${version} and ${fs.readdirSync(path.join(dst, "ports")).length} port entries`);
