#!/usr/bin/env node

const { runBinary } = require('./run-binary');

runBinary('vestige-mcp', 'vestige-mcp', process.argv.slice(2));
