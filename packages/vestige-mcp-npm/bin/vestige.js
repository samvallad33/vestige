#!/usr/bin/env node

const { runBinary } = require('./run-binary');

runBinary('vestige', 'vestige CLI', process.argv.slice(2));
