#!/usr/bin/env node

const { runBinary } = require('./run-binary');

runBinary('vestige-restore', 'vestige-restore', process.argv.slice(2));
