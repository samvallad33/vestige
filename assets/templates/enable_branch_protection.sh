#!/usr/bin/env bash
# One-click GitHub Branch Protection setup requiring Vestige Causal Gate & Shield checks on main.
# Usage: ./enable_branch_protection.sh <owner/repo> [branch]
set -euo pipefail

REPO="${1:?Usage: $0 <owner/repo> [branch]}"
BRANCH="${2:-main}"

gh api --method PUT \
  -H "Accept: application/vnd.github+json" \
  "/repos/${REPO}/branches/${BRANCH}/protection/required_status_checks" \
  --input - <<'EOF'
{
  "strict": true,
  "contexts": [
    "Vestige Causal Gate (Chesterton's Fence + Zombie Patch-ID)",
    "Vestige Concealment & Biba Integrity Shield"
  ]
}
EOF

echo "Required Vestige status checks enabled on ${REPO}@${BRANCH}."
