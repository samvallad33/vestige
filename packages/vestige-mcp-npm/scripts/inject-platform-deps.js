#!/usr/bin/env node
// #220: the committed package.json carries no optionalDependencies because
// the @vestige/mcp-* platform packages do not exist on the registry until
// the first platform-publish run, and pnpm's frozen-lockfile CI check
// fails on specifiers it cannot resolve. The release workflow runs this
// script immediately before publishing the meta package, after the
// platform packages for the same version are already published, so the
// published tarball pins them exactly (turbo-style) while the repository
// stays installable at every commit.
const fs = require('fs');
const path = require('path');

// Usage: node inject-platform-deps.js [package.json] (default: this package's own manifest)
const pkgPath = process.argv[2] || path.join(__dirname, '..', 'package.json');
const pkg = JSON.parse(fs.readFileSync(pkgPath, 'utf8'));
const version = pkg.version;

const PLATFORMS = [
  '@vestige/mcp-darwin-arm64',
  '@vestige/mcp-darwin-x64',
  '@vestige/mcp-linux-x64',
  '@vestige/mcp-linux-arm64',
  '@vestige/mcp-win32-x64',
];

pkg.optionalDependencies = Object.fromEntries(
  PLATFORMS.map((name) => [name, version]),
);
fs.writeFileSync(pkgPath, JSON.stringify(pkg, null, 2) + '\n');
console.log(`injected ${PLATFORMS.length} platform pins at ${version}`);
