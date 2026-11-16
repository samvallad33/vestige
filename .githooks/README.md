# Git hooks

Enable once per clone:

    git config core.hooksPath .githooks

`pre-commit` runs `scripts/hardcode_scan.sh` and blocks the commit on any
finding. CI runs the same scan plus its negative self-test
(`.github/workflows/hardcode-scan.yml`, job `hardcode-scan`).
