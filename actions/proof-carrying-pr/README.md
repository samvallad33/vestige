# Vestige proof-carrying PR

Fails a pull request when the tree at its head has no green test witness signed by a key you pin. It
builds `vestige-verify-proof` from this repository, reads the bundle and local git, and does not open a
Vestige store or reach the network beyond `git fetch`.

```yaml
- uses: actions/checkout@v4
  with:
    ref: ${{ github.event.pull_request.head.sha }}
    fetch-depth: 0
- uses: samvallad33/vestige/actions/proof-carrying-pr@main
```

Keys come from `.vestige/trusted-keys` and required commands from `.vestige/witness.toml`, both read
from the pull request's base branch. The `trusted-keys`, `require`, `strict`, `commit` and `base`
inputs add to them. The `verified` and `tree` outputs report the result.

The full guide is [docs/PROOF-CARRYING-PRS.md](../../docs/PROOF-CARRYING-PRS.md).
