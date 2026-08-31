#!/usr/bin/env bash
# check-runs-on-baseline.sh — run a built Linux binary inside the oldest distros
# Vestige claims to support, and assert it actually starts.
#
# WHY THIS EXISTS ALONGSIDE check-glibc-floor.sh
#
# The static check bounds symbol versions it knows to look for. That is a proxy,
# and the proxy has already been wrong twice:
#
#   - Its first version grepped only `GLIBC_`, so it reported a healthy 2.34
#     floor on a binary that still could not start on RHEL 9 because of
#     `GLIBCXX_3.4.30` — a whole symbol namespace it never looked at.
#   - A binary can also fail with no bad version at all, when it links a library
#     the target simply does not have: aarch64's `libmvec.so.1` first shipped in
#     glibc 2.38 and did not gain `_ZGVnN2v_logf` until 2.39.
#
# Running the binary is the only check that cannot be fooled by a symbol class
# nobody thought to enumerate. Issues #174 and #175 were exactly this failure:
# a binary that built and ran perfectly on the machine that produced it.
#
# USAGE
#   scripts/check-runs-on-baseline.sh target/<triple>/release/vestige-mcp ...
#   VESTIGE_BASELINE_IMAGES="rockylinux:9 debian:12" scripts/check-runs-on-baseline.sh ...
#
# Images default to the floor stated in README.md. Docker must be able to run
# the binary's architecture natively; this is meant for a runner whose arch
# matches the target it just built.
set -uo pipefail

IMAGES="${VESTIGE_BASELINE_IMAGES:-rockylinux:9 debian:12 ubuntu:22.04}"

if [ "$#" -eq 0 ]; then
  echo "usage: $0 <elf-binary> [elf-binary ...]" >&2
  exit 2
fi

if ! docker info >/dev/null 2>&1; then
  echo "SKIP: docker is not available, cannot run the baseline check." >&2
  echo "      This is a real gap in coverage, not a pass." >&2
  exit 0
fi

status=0

for bin in "$@"; do
  if [ ! -f "$bin" ]; then
    echo "FAIL: $bin does not exist" >&2
    status=1
    continue
  fi

  bin_dir="$(cd "$(dirname "$bin")" && pwd)"
  bin_name="$(basename "$bin")"

  for image in $IMAGES; do
    echo "== $bin_name on $image"
    output="$(docker run --rm -v "$bin_dir:/vestige-bin:ro" "$image" \
      "/vestige-bin/$bin_name" --version 2>&1)"
    rc=$?

    if [ "$rc" -eq 0 ]; then
      echo "   OK: $(printf '%s' "$output" | tail -n1)"
    else
      echo "FAIL: $bin_name does not start on $image (exit $rc)" >&2
      printf '%s\n' "$output" | sed 's/^/         /' >&2
      echo "       README.md promises this distro works. Either fix the binary's" >&2
      echo "       runtime requirements or correct the documented floor." >&2
      status=1
    fi
  done
done

if [ "$status" -eq 0 ]; then
  echo "OK: every binary starts on: $IMAGES"
fi

exit "$status"
