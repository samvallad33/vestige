#!/usr/bin/env bash
# The proof-carrying PR check. Builds vestige-verify-proof from this action's
# own checkout, unless VESTIGE_VERIFY_PROOF_BIN names a built one, then
# verifies the bundle for the head commit's tree against the keys and required
# commands on the base branch. Inputs arrive only as environment variables.
set -euo pipefail

workspace="${GITHUB_WORKSPACE:-$PWD}"
action_dir="${GITHUB_ACTION_PATH:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
source_root="$(cd "$action_dir/../.." && pwd)"

out() {
  if [ -n "${GITHUB_OUTPUT:-}" ]; then
    printf '%s=%s\n' "$1" "$2" >>"$GITHUB_OUTPUT"
  fi
}

bin="${VESTIGE_VERIFY_PROOF_BIN:-}"
if [ -z "$bin" ]; then
  echo "::group::Build vestige-verify-proof"
  cargo build --release --locked --quiet \
    --manifest-path "$source_root/crates/vestige-proof/Cargo.toml" \
    --bin vestige-verify-proof
  bin="$source_root/target/release/vestige-verify-proof"
  echo "::endgroup::"
fi

commit="${INPUT_COMMIT:-}"
if [ -z "$commit" ]; then commit="${PR_HEAD_SHA:-}"; fi
if [ -z "$commit" ]; then commit="${GITHUB_SHA:-HEAD}"; fi
if ! git -C "$workspace" cat-file -e "$commit^{commit}" 2>/dev/null; then
  git -C "$workspace" fetch --quiet --no-tags origin "$commit" || true
fi

# A bundle may also travel as a git note.
git -C "$workspace" fetch --quiet --no-tags origin \
  "+refs/notes/vestige-proof:refs/notes/vestige-proof" 2>/dev/null || true

args=(--repo "$workspace" --commit "$commit")

base="${INPUT_BASE:-}"
if [ -z "$base" ]; then base="${PR_BASE_REF:-${GITHUB_BASE_REF:-}}"; fi
if [ -n "$base" ]; then
  if git -C "$workspace" fetch --quiet --no-tags origin "$base" 2>/dev/null; then
    args+=(--base "$(git -C "$workspace" rev-parse FETCH_HEAD)")
  elif git -C "$workspace" rev-parse --verify --quiet "origin/$base^{commit}" >/dev/null; then
    args+=(--base "origin/$base")
  else
    args+=(--base "$base")
  fi
fi

while IFS= read -r key; do
  key="${key%%#*}"
  key="$(printf '%s' "$key" | tr -d '[:space:]')"
  if [ -n "$key" ]; then args+=(--trust "$key"); fi
done <<<"${INPUT_TRUSTED_KEYS:-}"

while IFS= read -r cmd; do
  if [ -n "${cmd//[[:space:]]/}" ]; then args+=(--require "$cmd"); fi
done <<<"${INPUT_REQUIRE:-}"

if [ "${INPUT_STRICT:-false}" = "true" ]; then args+=(--strict); fi

set +e
report="$("$bin" "${args[@]}" --format json)"
status=$?
set -e
if [ -n "${GITHUB_STEP_SUMMARY:-}" ]; then
  "$bin" "${args[@]}" --format text --summary "$GITHUB_STEP_SUMMARY" || true
else
  "$bin" "${args[@]}" --format text || true
fi
tree="$(printf '%s\n' "$report" | sed -n 's/^  "tree": "\([0-9a-f]*\)",$/\1/p' | head -n 1)"
out tree "$tree"
if [ "$status" -eq 0 ]; then
  out verified true
else
  out verified false
  echo "::error title=Proof-carrying PR::The tree at $commit has no verified green witness. See the job summary for each command."
fi
exit "$status"
