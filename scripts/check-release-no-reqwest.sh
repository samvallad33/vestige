#!/usr/bin/env bash
# Fail if any vestige-mcp feature set used by the release workflow links
# reqwest. Feature sets are read from .github/workflows/release.yml so a new
# target cannot ship an HTTP client without this check seeing it.
#
# `cargo tree -e features -i reqwest` exits 0 and prints the crate when
# reqwest is in the graph. It exits non-zero with "did not match any
# packages" when the crate is absent. Any other failure is a failed check.
set -euo pipefail

root=$(cd "$(dirname "$0")/.." && pwd)
workflow="$root/.github/workflows/release.yml"
cd "$root"

mapfile -t rows < <(python3 - "$workflow" <<'PY'
import re
import sys

text = open(sys.argv[1], encoding="utf-8").read()
joined = []
pending = ""
for raw in text.splitlines():
    if raw.rstrip().endswith("\\"):
        pending += raw.rstrip()[:-1] + " "
    else:
        joined.append(pending + raw)
        pending = ""
if pending:
    joined.append(pending)

sets = []
target = ""
for line in joined:
    target_match = re.search(r"(?:^|\s)target:\s*([A-Za-z0-9_.:-]+)\s*$", line)
    if target_match:
        target = target_match.group(1)
    flags_match = re.search(r"""cargo_flags:\s*"([^"]*)"$""", line)
    if flags_match:
        sets.append((target or "cargo_flags", flags_match.group(1).strip()))

for line in joined:
    if "vestige-mcp" not in line:
        continue
    if "matrix.cargo_flags" in line:
        continue
    if not re.search(r"\bcargo\s+(?:ndk\b.*\s)?build\b", line):
        continue
    args = []
    if "--no-default-features" in line:
        args.append("--no-default-features")
    features = re.search(r"--features\s+(\S+)", line)
    if features:
        args.extend(["--features", features.group(1).strip("\"'")])
    target_flag = re.search(r"--target\s+(\S+)", line)
    if target_flag:
        label = target_flag.group(1)
    elif "cargo ndk" in line or "linux-android" in line:
        label = "aarch64-linux-android"
    else:
        label = "vestige-mcp"
    sets.append((label, " ".join(args)))

if not sets:
    sys.stderr.write("error: no release feature sets found in release.yml\n")
    sys.exit(2)

seen = set()
for label, args in sets:
    key = args
    if key in seen:
        continue
    seen.add(key)
    sys.stdout.write(f"{label}\t{args}\n")
PY
)

if [[ ${#rows[@]} -eq 0 ]]; then
  echo "error: release.yml produced no feature sets" >&2
  exit 2
fi

fail=0
for row in "${rows[@]}"; do
  label=${row%%$'\t'*}
  args=${row#*$'\t'}
  echo "== $label${args:+ ($args)}"
  set +e
  output=$(cargo tree -p vestige-mcp --locked -e features -i reqwest $args 2>&1)
  status=$?
  set -e
  if [[ $status -eq 0 ]] || grep -Eq 'reqwest v[0-9]' <<<"$output"; then
    echo "FAIL: $label links reqwest" >&2
    printf '%s\n' "$output" >&2
    fail=1
    continue
  fi
  if ! grep -q 'did not match any packages' <<<"$output"; then
    echo "FAIL: cargo tree did not prove reqwest absent for $label (exit $status)" >&2
    printf '%s\n' "$output" >&2
    fail=1
    continue
  fi
  echo "ok: reqwest absent"
done

exit "$fail"
