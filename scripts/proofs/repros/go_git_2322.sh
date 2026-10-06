#!/bin/sh
# go-git #2322: PlainClone of a repo that contains prn.sh.
# Exit 0 when the clone succeeds. Exit 1 when the path is rejected.
# Exit 125 when this tree cannot be built. The module cache lives outside
# the worktree so git clean does not throw it away.
set -eu
root=$(pwd)
if [ ! -f "$root/go.mod" ]; then
  echo "no go.mod" >&2
  exit 125
fi
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
cd "$work"
cat > go.mod <<EOF
module repro.local
go 1.22

require github.com/go-git/go-git/v5 v5.19.1

replace github.com/go-git/go-git/v5 => $root
EOF
cat > main.go <<'EOF'
package main

import (
	"fmt"
	"os"
	"os/exec"
	"strings"

	git "github.com/go-git/go-git/v5"
)

func run(dir string, args ...string) error {
	cmd := exec.Command("git", args...)
	cmd.Dir = dir
	cmd.Env = append(os.Environ(),
		"GIT_AUTHOR_NAME=t", "GIT_AUTHOR_EMAIL=t@t.com",
		"GIT_COMMITTER_NAME=t", "GIT_COMMITTER_EMAIL=t@t.com")
	out, err := cmd.CombinedOutput()
	if err != nil {
		return fmt.Errorf("%s: %s", err, out)
	}
	return nil
}

func main() {
	src, err := os.MkdirTemp("", "src")
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(125)
	}
	defer os.RemoveAll(src)
	if err = run(src, "init", "-q", "-b", "master"); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(125)
	}
	if err = os.WriteFile(src+"/prn.sh", []byte("echo hi\n"), 0o644); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(125)
	}
	if err = run(src, "add", "-A"); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(125)
	}
	if err = run(src, "commit", "-q", "-m", "add prn.sh"); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(125)
	}
	dst, err := os.MkdirTemp("", "dst")
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(125)
	}
	defer os.RemoveAll(dst)
	_, err = git.PlainClone(dst, false, &git.CloneOptions{URL: src})
	if err == nil {
		os.Exit(0)
	}
	msg := err.Error()
	fmt.Fprintln(os.Stderr, msg)
	if strings.Contains(msg, "invalid path") {
		os.Exit(1)
	}
	os.Exit(1)
}
EOF
# The repo's go.mod asks for 1.25. The repro module must run that toolchain
# or `go run` stops before it compiles.
export GOTOOLCHAIN="${GOTOOLCHAIN:-go1.25.4}"
export GOFLAGS="${GOFLAGS:--mod=mod}"
if ! go run . >"$work/out" 2>"$work/err"; then
  cat "$work/err" >&2
  if grep -q "invalid path" "$work/err"; then
    exit 1
  fi
  # A compile or module failure is not the bug.
  if grep -Eq "go: |undefined:|build failed|no required module" "$work/err"; then
    exit 125
  fi
  exit 1
fi
exit 0
