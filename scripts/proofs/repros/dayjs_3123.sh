#!/bin/sh
# dayjs #3123. The assertion is the one in the issue: on a US DST fallback
# date, dayjs.tz(ms, 'UTC').startOf('day') equals UTC midnight.
# Exit 0 when it does, 1 when the host DST transition leaks in, 125 when this
# tree has no timezone plugin to load.
#
# The bundle is esbuild, not `npm run build`. Rollup on the 2020 revisions
# does not run under Node 22, and a checkout must not be left dirty.
set -eu
export TZ=America/Los_Angeles
root=$(pwd)
if [ ! -f "$root/src/index.js" ] || [ ! -f "$root/src/plugin/utc/index.js" ] || [ ! -f "$root/src/plugin/timezone/index.js" ]; then
  echo "timezone plugin is not in this tree" >&2
  exit 125
fi
entry=$(mktemp /tmp/dayjs-entry.XXXXXX.mjs)
out=$(mktemp /tmp/dayjs-bundle.XXXXXX.js)
trap 'rm -f "$entry" "$out"' EXIT
cat > "$entry" <<EOF
import dayjs from "$root/src/index.js";
import utc from "$root/src/plugin/utc/index.js";
import timezone from "$root/src/plugin/timezone/index.js";
dayjs.extend(utc);
dayjs.extend(timezone);
const ms = new Date("2024-11-03T23:00:00Z").getTime();
const expected = new Date("2024-11-03T00:00:00Z").getTime();
let buggy;
try {
  buggy = dayjs.tz(ms, "UTC").startOf("day").valueOf();
} catch (err) {
  console.error(err);
  process.exit(125);
}
console.log("got", new Date(buggy).toISOString());
process.exit(buggy === expected ? 0 : 1);
EOF
if ! npx --offline --yes esbuild@0.25.0 "$entry" --bundle --platform=node --format=cjs --outfile="$out" >/dev/null 2>&1; then
  echo "esbuild failed" >&2
  exit 125
fi
node "$out"
