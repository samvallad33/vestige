#!/usr/bin/env bash
# check-glibc-floor.sh — Fail if a Linux release binary raises its runtime floor.
#
# WHY THIS EXISTS
#
# v2.3.0 shipped an x86_64-unknown-linux-gnu binary that aborted at startup on
# every distro older than Ubuntu 24.04:
#
#   ./vestige-mcp: /lib/x86_64-linux-gnu/libc.so.6: version `GLIBC_2.38' not found
#
# The server died before answering a single MCP request, which is what issues
# #174 and #175 actually reported — the "spec conformance violations" in those
# reports were a harness watching an already-dead process, not protocol defects.
#
# Three separate symbol classes have each pushed the floor up, and each one was
# invisible in a normal build and test run:
#
#   1. __isoc23_strtol/strtoll/strtoull at GLIBC_2.38, from the prebuilt
#      libonnxruntime.a that `ort-sys` links statically.
#   2. pidfd_getpid/pidfd_spawnp at GLIBC_2.39, from Rust std's `weak!` macro in
#      the `vestige` CLI. The symbols are weak; the version need is NOT, and the
#      loader refuses to start regardless of the symbol's binding.
#   3. libstdc++'s GLIBCXX_/CXXABI_ versions, set by whichever GCC links the
#      build. These do not contain the string "GLIBC_" and so were missed
#      entirely by the first version of this script.
#
# crates/vestige-mcp/src/glibc_compat.rs handles (1) and (2). This script proves
# they stayed gone and bounds (3).
#
# WHAT IT CHECKS, per binary
#
#   - No `__isoc23_*` symbol is imported.
#   - Highest GLIBC_x.y   requirement <= VESTIGE_MAX_GLIBC   (default 2.34)
#   - Highest GLIBCXX_x.y requirement <= VESTIGE_MAX_GLIBCXX (default 3.4.29)
#   - Highest CXXABI_x.y  requirement <= VESTIGE_MAX_CXXABI  (default 1.3.13)
#
# The defaults are the ceilings measured on Rocky Linux 9.3, the oldest distro
# README.md promises: glibc 2.34, and libstdc++.so.6.0.29 providing at most
# GLIBCXX_3.4.29 and CXXABI_1.3.13. Anything above those will not start there.
#
# Release Linux builds run inside AlmaLinux 9 with gcc-toolset-14 precisely so
# these ceilings hold; building on the Ubuntu 24.04 runner instead binds
# GLIBCXX_3.4.30 / CXXABI_1.3.15 and silently drops RHEL 9 support. Raising a
# default here means editing README.md in the same commit.
#
# This is a static check. `scripts/check-runs-on-baseline.sh` is the unfoolable
# companion: it runs the binary inside those distro images.
#
# USAGE
#   scripts/check-glibc-floor.sh target/<triple>/release/vestige-mcp ...
#   VESTIGE_MAX_GLIBC=2.28 scripts/check-glibc-floor.sh path/to/binary
#
# Requires binutils (readelf, nm), which GitHub's ubuntu runners ship by default.
set -uo pipefail

MAX_GLIBC="${VESTIGE_MAX_GLIBC:-2.34}"
MAX_GLIBCXX="${VESTIGE_MAX_GLIBCXX:-3.4.29}"
MAX_CXXABI="${VESTIGE_MAX_CXXABI:-1.3.13}"

# Shared libraries present on every distro at the documented floor. libmvec.so.1
# is deliberately absent: usearch's `-ffast-math` used to drag it in through
# auto-vectorized std::log in a metric Vestige never calls, and the release
# workflow now compiles with -U__FAST_MATH__ so no target needs it at all.
BASELINE_LIBS="${VESTIGE_BASELINE_LIBS:-ld-linux-aarch64.so.1 ld-linux-x86-64.so.2 libc.so.6 libm.so.6 libgcc_s.so.1 libstdc++.so.6 libdl.so.2 libpthread.so.0 librt.so.1}"

if [ "$#" -eq 0 ]; then
  echo "usage: $0 <elf-binary> [elf-binary ...]" >&2
  exit 2
fi

for tool in readelf nm; do
  command -v "$tool" >/dev/null 2>&1 || {
    echo "FAIL: '$tool' not found. Install binutils to run this check." >&2
    exit 2
  }
done

status=0

# True when $1 sorts strictly later than $2 under version ordering.
version_gt() {
  [ "$1" != "$2" ] && [ "$(printf '%s\n%s\n' "$1" "$2" | sort -V | tail -n1)" = "$1" ]
}

# Bound one versioned-symbol namespace (GLIBC_, GLIBCXX_, CXXABI_) for one
# binary. `GLIBC_` cannot match `GLIBCXX_`, since the character after "GLIBC"
# differs, so the three namespaces stay disjoint.
check_namespace() {
  local bin="$1" prefix="$2" max="$3"
  local versions max_found

  versions="$(readelf -V "$bin" 2>/dev/null \
    | grep -oE "${prefix}[0-9]+(\.[0-9]+)*" | sed "s/^${prefix}//" | sort -uV)"
  [ -z "$versions" ] && return 0

  max_found="$(printf '%s\n' "$versions" | tail -n1)"
  echo "   highest ${prefix%_} requirement: $max_found (allowed: <= $max)"
  version_gt "$max_found" "$max" || return 0

  echo "FAIL: $bin requires ${prefix}${max_found}, above the $max ceiling." >&2
  echo "       Distros providing at most ${prefix}${max} will refuse to start it," >&2
  echo "       exactly like the v2.3.0 regression in issues #174 and #175." >&2

  # Attribute the requirement to the library that demands it: "GLIBC_2.39 from
  # libmvec.so.1" is a completely different fix from "from libc.so.6".
  echo "       Required by:" >&2
  readelf -V "$bin" 2>/dev/null \
    | awk -v want="${prefix}${max_found}" '
        /File: /   { lib = $0; sub(/.*File: /, "", lib); sub(/ .*/, "", lib) }
        index($0, "Name: " want) { print "         " lib }
      ' | sort -u >&2

  # Match symbols by pattern, not by column: readelf pads the name column
  # differently for unnamed entries and a positional field prints "UND".
  local offenders
  offenders="$(readelf --dyn-syms --wide "$bin" 2>/dev/null \
    | grep -oE "[A-Za-z_][A-Za-z0-9_]*@${prefix}${max_found}" | sort -u)"
  if [ -n "$offenders" ]; then
    echo "       Symbols pulling it in:" >&2
    printf '%s\n' "$offenders" | sed 's/^/         /' >&2
  else
    echo "       No symbol carries it: it comes from a library version-need" >&2
    echo "       record, not a symbol reference." >&2
  fi
  status=1
}

for bin in "$@"; do
  if [ ! -f "$bin" ]; then
    echo "FAIL: $bin does not exist" >&2
    status=1
    continue
  fi

  if [ "$(head -c 4 "$bin" | od -An -tx1 | tr -d ' \n')" != "7f454c46" ]; then
    echo "FAIL: $bin is not an ELF binary" >&2
    status=1
    continue
  fi

  echo "== $bin"

  # A library the loader cannot find is as fatal as a missing symbol version, and
  # it fails earlier and differently: aarch64's libmvec.so.1 first shipped in
  # glibc 2.38 and did not gain _ZGVnN2v_logf until 2.39, so an arm64 binary
  # linking it dies with "cannot open shared object file" on Debian 12, Ubuntu
  # 22.04 and RHEL 9 no matter what its GLIBC_ versions say. An allowlist catches
  # that even when every version number looks fine.
  needed="$(readelf -d "$bin" 2>/dev/null \
    | sed -n 's/.*Shared library: \[\(.*\)\]/\1/p' | sort)"
  echo "   needs: $(printf '%s' "$needed" | tr '\n' ' ')"

  for lib in $needed; do
    case " $BASELINE_LIBS " in
      *" $lib "*) ;;
      *)
        echo "FAIL: $bin has DT_NEEDED on $lib, which is not guaranteed to exist" >&2
        echo "       on the baseline distros. If it is genuinely always present," >&2
        echo "       add it to BASELINE_LIBS with the evidence; otherwise stop" >&2
        echo "       linking it." >&2
        status=1
        ;;
    esac
  done

  isoc23="$(nm -D --undefined-only "$bin" 2>/dev/null | grep -oE '__isoc23_[a-z_]+' | sort -u)"
  if [ -n "$isoc23" ]; then
    echo "FAIL: imports C23 symbols that require glibc >= 2.38:" >&2
    printf '%s\n' "$isoc23" | sed 's/^/         /' >&2
    echo "       Add the missing forwarders to crates/vestige-mcp/src/glibc_compat.rs" >&2
    echo "       and make sure the module is wired into every binary root." >&2
    status=1
  fi

  check_namespace "$bin" "GLIBC_" "$MAX_GLIBC"
  check_namespace "$bin" "GLIBCXX_" "$MAX_GLIBCXX"
  check_namespace "$bin" "CXXABI_" "$MAX_CXXABI"
done

if [ "$status" -eq 0 ]; then
  echo "OK: floors hold (GLIBC <= $MAX_GLIBC, GLIBCXX <= $MAX_GLIBCXX, CXXABI <= $MAX_CXXABI)"
fi

exit "$status"
