#!/bin/sh
# winston #800: File transport rotation with zippedArchive + tailable + maxsize.
# Exit 0 when a burst of lines does not raise ENOENT.
# Exit 1 when it does. Exit 125 when this tree has no File transport to load.
# node_modules is gitignored, so a previous commit's install can survive
# `git clean`. Reinstall when package.json changes.
set -eu
if [ ! -f package.json ]; then
  echo "no package.json" >&2
  exit 125
fi
stamp=$(sha256sum package.json package-lock.json 2>/dev/null | sha256sum | awk '{print $1}')
if [ ! -d node_modules/async ] && [ ! -d node_modules/winston ] || [ "$(cat node_modules/.proof-stamp 2>/dev/null || true)" != "$stamp" ]; then
  if ! npm install --no-audit --no-fund --silent; then
    echo "npm install failed" >&2
    exit 125
  fi
  printf '%s\n' "$stamp" > node_modules/.proof-stamp
fi
git checkout -q -- package.json package-lock.json 2>/dev/null || true
node <<'EOF'
const fs = require("fs");
const os = require("os");
const path = require("path");
const root = process.cwd();
let winston;
try {
  winston = require(root);
} catch (err) {
  try {
    winston = require(path.join(root, "lib/winston"));
  } catch (err2) {
    console.error(err2);
    process.exit(125);
  }
}
const dir = fs.mkdtempSync(path.join(os.tmpdir(), "winston-800-"));
const filename = path.join(dir, "server.log");
const File = winston.transports && winston.transports.File;
if (!File) {
  console.error("no File transport");
  process.exit(125);
}
let logger;
try {
  if (typeof winston.Logger === "function" || typeof winston.createLogger === "function") {
    const create = winston.createLogger || function (opts) { return new winston.Logger(opts); };
    logger = create.call(winston, {
      transports: [new File({
        filename,
        json: false,
        maxsize: 80,
        maxFiles: 3,
        tailable: true,
        zippedArchive: true,
      })],
    });
  } else if (typeof winston.add === "function") {
    winston.add(File, {
      filename,
      json: false,
      maxsize: 80,
      maxFiles: 3,
      tailable: true,
      zippedArchive: true,
    });
    logger = winston;
  } else {
    console.error("unrecognized winston API");
    process.exit(125);
  }
} catch (err) {
  console.error(err);
  process.exit(125);
}
let enoent = false;
process.on("uncaughtException", (err) => {
  if (err && (err.code === "ENOENT" || /ENOENT/.test(String(err)))) enoent = true;
});
process.on("unhandledRejection", (err) => {
  if (err && (err.code === "ENOENT" || /ENOENT/.test(String(err)))) enoent = true;
});
if (typeof logger.on === "function") logger.on("error", (err) => {
  if (err && (err.code === "ENOENT" || /ENOENT/.test(String(err)))) enoent = true;
});
const line = "x".repeat(60);
for (let i = 0; i < 2000; i++) {
  try {
    logger.log("info", line + " " + i);
  } catch (err) {
    if (err && (err.code === "ENOENT" || /ENOENT/.test(String(err)))) enoent = true;
  }
}
setTimeout(() => {
  try { fs.rmSync(dir, { recursive: true, force: true }); } catch (e) {}
  process.exit(enoent ? 1 : 0);
}, 1000);
EOF
