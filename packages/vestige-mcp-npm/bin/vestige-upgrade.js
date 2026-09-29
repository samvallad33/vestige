#!/usr/bin/env node

const { runBinary } = require('./run-binary');

runBinary('vestige-upgrade', 'vestige-upgrade', process.argv.slice(2));
