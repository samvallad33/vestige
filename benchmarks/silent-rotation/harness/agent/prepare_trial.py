#!/usr/bin/env python3
"""Per-trial randomizer for the N-trial Silent Rotation benchmark.

Makes each trial information-theoretically INDEPENDENT so a skeptic cannot claim
the agents were tuned to one answer. Per trial it:
  1. Picks a random set of 3 neutral key names + a random CORRECT one among them
     (names are randomized too, so there is no fixed "borealis is always right").
  2. Generates fresh random 32-byte hex for all 3 keys.
  3. Rewrites shared/src/keyring.ts (+ .repo-snapshot copy) with the new keys.
  4. Regenerates the production corpus: fresh tokens signed under the TRIAL's
     correct key material, written to a per-trial PROD_CORPUS path.
  5. Re-templates the Vestige seed script so the memory names the TRIAL's correct
     key + its decoys (the deciding fact stays ONLY in memory, never the repo).
  6. Leaves configs blank (RED base) and returns a manifest describing the trial,
     so the run is fully reproducible from the master seed.

Everything here is deterministic given a seed, so `--seed S` reproduces an exact
trial. This is the "reproduce this exact trial" affordance a skeptic needs.

Usage (called by the experiment loop, or standalone for offline checks):
  python3 prepare_trial.py --repo <torture-v3.5> --trial 3 --master-seed 12345 \
      --corpus-out <path.json> --manifest-out <path.json>
Prints the chosen correct kid to stdout (last line) for the caller to export.
"""
from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import os
import random
import re
import secrets
from pathlib import Path

# Neutral, disposition-free name pool (stars, constellations, minerals). No name
# implies "active/prod/current". Randomizing the NAMES (not just which is correct)
# kills any "the middle one is always right" prior. Pool must be >= KEYRING_SIZE.
# A LARGE keyring is what makes the benchmark un-riggable: with K keys, a
# memoryless agent's blind guess is correct only 1/K of the time, so the control
# fails ~always and the null-hypothesis p-value is crushing (1/K per correct pick).
NAME_POOL = [
    "atlas", "borealis", "cirrus", "dorado", "electra", "fornax", "helios",
    "indus", "juno", "kepler", "lyra", "mensa", "norma", "orion", "pavo",
    "quasar", "rigel", "sirius", "tucana", "vega", "wren", "zephyr",
    "altair", "antares", "arcturus", "bellatrix", "capella", "castor", "deneb",
    "draco", "fomalhaut", "gemini", "hydra", "izar", "lyric", "mizar", "nashira",
    "octans", "perseus", "polaris", "procyon", "regulus", "spica", "tarazed",
    "umbriel", "vela", "wezen", "yildun", "aludra", "basalt", "cobalt", "dunite",
    "feldspar", "garnet", "hematite", "jasper", "kyanite", "olivine", "pyrite",
    "quartz", "rutile", "schist", "talc", "zircon",
]

# How many keys live in the keyring per trial (1 correct + KEYRING_SIZE-1 decoys).
# 50 makes blind-guess success 2% -> the control cannot 'luck into' the answer,
# and 5 clean sync trials give a null p-value of (1/50)^5 ~ 3e-9. Override via env.
KEYRING_SIZE = int(os.environ.get("KEYRING_SIZE", "50"))

FAR_FUTURE_EXP = 4102444800  # year 2100 — tokens never expire (matches verify.ts)
ISSUER = "identity.meridian.internal"
CORPUS_SIZE = 5


def b64url(raw: bytes) -> str:
    import base64
    return base64.urlsafe_b64encode(raw).decode().rstrip("=")


def canonical(claims: dict) -> str:
    # MUST match torture-v3.5 canonical.ts field order: sub, plan, iat, exp.
    return "\n".join(f"{k}={claims[k]}" for k in ("sub", "plan", "iat", "exp"))


def sign_token(claims: dict, key_hex: str) -> str:
    key = bytes.fromhex(key_hex)
    header = {"alg": "HS256", "typ": "JWT", "iss": ISSUER}
    h = b64url(json.dumps(header).encode())
    p = b64url(json.dumps(claims).encode())
    sig = b64url(hmac.new(key, canonical(claims).encode(), hashlib.sha256).digest())
    return f"{h}.{p}.{sig}"


def make_keyring_ts(keys: dict[str, str]) -> str:
    lines = "\n".join(f'  k_{n}: "{hexv}",' for n, hexv in keys.items())
    return (
        "/**\n"
        " * @meridian/shared — the platform signing keyring.\n"
        " *\n"
        " * The keyring holds the HMAC signing keys the platform can use to sign and\n"
        " * verify session tokens. Keys are referenced by a short key id (`kid`). Each\n"
        " * service selects which key it uses from its own runtime config; the keyring\n"
        " * itself is just the material.\n"
        " *\n"
        " * Keys are 32-byte values, hex-encoded. (Dev material, committed for the local\n"
        " * harness; injected from the KMS in production.)\n"
        " */\n"
        "export const KEYRING: Record<string, string> = {\n"
        "  // Each entry is a provisioned HMAC key. Presence in the keyring does not imply\n"
        "  // a service currently selects it; selection is per-service runtime config.\n"
        f"{lines}\n"
        "};\n\n"
        "/** Resolve a key id to its raw signing material, or throw if the id is unknown. */\n"
        "export function keyForId(kid: string): Buffer {\n"
        "  const hex = KEYRING[kid];\n"
        "  if (!hex) {\n"
        "    throw new Error(`unknown key id: ${JSON.stringify(kid)}`);\n"
        "  }\n"
        "  return Buffer.from(hex, \"hex\");\n"
        "}\n"
    )


def make_seed_sh(correct_kid: str, decoys: list[str], vestige_bin_default: str,
                 strip_cause_edge: bool = False) -> str:
    d0, d1 = decoys[0], decoys[1]
    # The memory names the TRIAL's correct kid + decoys. This is the ONLY place
    # the answer exists; it lives in the Vestige DB, never in the agent checkout.
    #
    # CORPUS CONSTRUCTION (verified offline, no paid API, Jul 17 2026, red-team gated):
    # The CAUSE is a NATURAL rotation runbook written in operational/ownership
    # vocabulary that genuinely does NOT reuse the crash's words (charge / 500 /
    # verify / reject / signature). It shares the `active_key` entity with the
    # FAILURE (that is backfill's causal join key) and names the correct kid.
    # Around it sit 5 REALISTIC same-neighborhood distractors: real ops incidents
    # that legitimately discuss signing/verify/500/charge but are NOT the cause
    # (none tagged `active_key`). A real production history contains exactly these.
    #
    # Consequence, proven on real nomic-embed-text embeddings: for the failure
    # query, pure cosine ranks the 5 distractors ABOVE the cause (cause at rank
    # ~#6 of 8), so a FAIR top-k RAG (k=3 or k=5) MISSES the cause and fails
    # honestly -- not because it was crippled, but because the cause is genuinely
    # semantically distant. Meanwhile `vestige backfill` reaches the cause via the
    # `active_key` join at a high similarity_rank, and prints a receipt proving a
    # vector search of the same budget could not have surfaced it. Both halves
    # hold simultaneously with nothing strawmanned. See cosine_probe.py gate.
    cause = (
        f"Rotation runbook, Q3: we migrated the live signer to {correct_kid} and let the old "
        f"material age out. Every credential minted since the cutover carries the new fingerprint. "
        f"{d0} was the launch signer and is now decommissioned; {d1} sits pre-provisioned for a "
        f"scheduled future cutover that has not happened. Anything consuming already-minted "
        f"credentials must trust {correct_kid} or it will turn away the entire installed base."
    )
    # 5 real distractors that genuinely live in the symptom's neighborhood
    # (signing / verify / 500 / charge), none of which is the cause. Textually
    # closer to the failure than the cause is -- by construction of a real history.
    d_ops = (
        f"Closed ticket OPS-522: a staging box failed token verification after a hand-edited "
        f"config; on-call reset the key id to {d0} and verification recovered. Staging-only, "
        f"nothing rotated."
    )
    d_verify = (
        "Ledger verifier threw a 500 on charge when the token had no 'kid' header; we added a "
        "null-check so a malformed token returns 401 instead of crashing the charge path."
    )
    d_issuer = (
        "Identity issuer intermittently returned 500 during token minting under load; root cause "
        "was a connection-pool timeout to the signing HSM, fixed by raising the pool size. No key "
        "change."
    )
    d_alert = (
        "Observability: added a Grafana dashboard and a PagerDuty alert on the checkout "
        "error-rate SLO. When 5xx responses on the billing endpoint cross 2 percent over five "
        "minutes the on-call engineer is paged. Pure monitoring change, no request path touched."
    )
    d_schema = (
        "Externalize-secrets refactor: identity and ledger now read the active signing key id "
        "from env instead of a hardcoded default; the schema requires the key id field to be "
        "present at boot."
    )
    failure = (
        "PRODUCTION OUTAGE: after the externalize-secrets refactor, the identity issuer and the "
        "ledger verifier both come up with no signing key id selected, so every charge attempt "
        "throws and the charge path returns a generic 500. No user can be billed. The keyring "
        "still has all keys; nothing records which key id production was actually issuing tokens "
        "under."
    )
    return f"""#!/usr/bin/env bash
# AUTO-GENERATED per-trial seed. The correct key ({correct_kid}) lives ONLY here,
# in the Vestige memory — never in the agent's checkout.
set -euo pipefail
VESTIGE_BIN="${{VESTIGE_BIN:-{vestige_bin_default}}}"
DATA_DIR="${{VESTIGE_DATA_DIR:-$(cd "$(dirname "$0")" && pwd)/.vestige-demo-db}}"
if [[ ! -x "$VESTIGE_BIN" ]]; then echo "vestige binary not found at $VESTIGE_BIN" >&2; exit 1; fi
rm -rf "$DATA_DIR"; mkdir -p "$DATA_DIR"
V() {{ "$VESTIGE_BIN" --data-dir "$DATA_DIR" "$@"; }}
# CAUSE — natural runbook, distant vocabulary, {"EDGE STRIPPED (ablation: no shared active_key tag)" if strip_cause_edge else "shares active_key"}, names {correct_kid}
V ingest {json.dumps(cause)} --node-type decision --ago-days 4 --tags {json.dumps("keyring_rotation" if strip_cause_edge else "keyring_rotation,active_key")} --source "shared/src/keyring.ts"
# 5 realistic same-neighborhood distractors (NOT the cause, none tagged active_key)
V ingest {json.dumps(d_ops)} --node-type event --ago-days 12 --tags "ops,staging,config" --source "ops/runbook"
V ingest {json.dumps(d_verify)} --node-type event --ago-days 9 --tags "ledger_service,verify" --source "ledger-service/src/verify.ts"
V ingest {json.dumps(d_issuer)} --node-type event --ago-days 7 --tags "identity_service,hsm" --source "identity-service/src/issuer.ts"
V ingest {json.dumps(d_alert)} --node-type event --ago-days 5 --tags "ledger_service,alerting" --source "ledger-service/src/server.ts"
V ingest {json.dumps(d_schema)} --node-type event --ago-days 6 --tags "refactor,config_schema" --source "shared/src/config.ts"
# 2 pure noise events
V ingest "Gateway-Service: raised the per-plan rate limits (free 10, pro 100, enterprise 1000) and moved the counters to an in-memory map. Edge stays stateless." --node-type event --ago-days 3 --tags "gateway_service,rate_limit" --source "gateway-service/src/index.ts"
V ingest "Ledger-Service: refactored monthly statement generation to stream pages instead of buffering the whole document. Cut peak memory on large statements." --node-type event --ago-days 2 --tags "ledger_service,statements,pdf"
# FAILURE — the production outage (shares active_key with the cause)
V ingest {json.dumps(failure)} --node-type event --ago-days 0 --tags "active_key,crash" --source "ledger-service/src/server.ts"
"""


def prepare_trial(repo: Path, trial: int, master_seed: int, corpus_out: Path,
                  manifest_out: Path, vestige_bin: str) -> str:
    # Deterministic RNG per (master_seed, trial) so a trial is exactly reproducible.
    rng = random.Random(f"{master_seed}:{trial}")

    # 1. GUARANTEE a DISTINCT correct key per trial: shuffle the whole name pool
    #    once from the master seed, and assign trial N its own correct name (the
    #    Nth in the shuffled order). This makes the per-trial correct keys VISIBLY
    #    different across the experiment table (no "it's always k_indus"), while
    #    staying fully reproducible from the master seed. The two DECOYS are drawn
    #    randomly per trial from the remaining pool.
    pool_order = list(NAME_POOL)
    random.Random(f"{master_seed}:correct-key-order").shuffle(pool_order)
    correct_name = pool_order[(trial - 1) % len(pool_order)]
    decoy_pool = [n for n in NAME_POOL if n != correct_name]
    n_decoys = max(1, min(KEYRING_SIZE, len(NAME_POOL)) - 1)
    decoy_names = rng.sample(decoy_pool, n_decoys)
    names = [correct_name] + decoy_names
    rng.shuffle(names)  # keyring order isn't correct-key-first (no positional tell)
    correct_kid = f"k_{correct_name}"
    decoy_kids = [f"k_{n}" for n in decoy_names]

    # 2. Fresh random 32-byte hex per key (seeded from the trial RNG for repro).
    keys = {n: rng.getrandbits(256).to_bytes(32, "big").hex() for n in names}

    # 3. Rewrite keyring.ts in the live repo AND the snapshot.
    keyring_ts = make_keyring_ts(keys)
    (repo / "shared/src/keyring.ts").write_text(keyring_ts)
    (repo / ".repo-snapshot/shared/src/keyring.ts").write_text(keyring_ts)

    # 4. Regenerate the production corpus under the TRIAL's correct key.
    correct_hex = keys[correct_name]
    plans = ["enterprise", "pro", "free"]
    tokens = []
    for i in range(CORPUS_SIZE):
        claims = {"sub": f"user_{1001+i}", "plan": plans[i % len(plans)],
                  "iat": 1700000000 + i * 100, "exp": FAR_FUTURE_EXP}
        tokens.append({"sub": claims["sub"], "plan": claims["plan"],
                       "token": sign_token(claims, correct_hex)})
    corpus = {
        "note": "Already-issued PRODUCTION tokens, signed under this trial's live "
                "key. Held in the harness, in NO checkout.",
        "signed_under": correct_kid,
        "tokens": tokens,
    }
    corpus_out.parent.mkdir(parents=True, exist_ok=True)
    corpus_out.write_text(json.dumps(corpus, indent=2))

    # 5. Re-template the seed so memory names THIS trial's correct kid + decoys.
    #    CRITICAL: write it to BOTH the working repo AND the snapshot. run-sync.sh
    #    seeds the DB from the working repo's seed, but then calls reset-repo.sh
    #    which restores the working tree FROM the snapshot -- if the snapshot's
    #    seed is stale (old key), a later re-seed uses the wrong key and the whole
    #    trial silently seeds memory with a stale key (backfill then surfaces the
    #    wrong kid -> sync ships the wrong key -> failed_still_red). Keeping both in
    #    sync (like the keyring + configs above) is mandatory, not optional.
    seed_sh = make_seed_sh(correct_kid, decoy_kids, vestige_bin)
    (repo / ".vestige-seed.sh").write_text(seed_sh)
    snap_seed = repo / ".repo-snapshot" / ".vestige-seed.sh"
    if snap_seed.parent.exists():
        snap_seed.write_text(seed_sh)

    # ABLATION VARIANT (PREREGISTRATION.md, sync-noedge arm): byte-identical
    # corpus except the cause's shared `active_key` tag is stripped, so the
    # hand-authored causal edge does not exist. Written EVERY trial, from the
    # SAME deterministic content, so the noedge arm runs on the same trial set
    # with exactly one variable changed. run-sync-noedge.sh seeds from this
    # file; every other arm seeds from .vestige-seed.sh.
    seed_noedge = make_seed_sh(correct_kid, decoy_kids, vestige_bin,
                               strip_cause_edge=True)
    (repo / ".vestige-seed-noedge.sh").write_text(seed_noedge)
    snap_noedge = repo / ".repo-snapshot" / ".vestige-seed-noedge.sh"
    if snap_noedge.parent.exists():
        snap_noedge.write_text(seed_noedge)

    # 6. Ensure configs are BLANK (RED base). Neutralize decoy-hint comments so a
    #    fixed decoy name isn't baked in (the hints named atlas/cirrus originally).
    for rel, field in [("identity-service/src/config.ts", "activeKid"),
                       ("ledger-service/src/config.ts", "trustedKid")]:
        for base in (repo, repo / ".repo-snapshot"):
            p = base / rel
            t = p.read_text()
            t = re.sub(rf'({field}\s*:\s*)"[^"]*"', r'\g<1>""', t)
            # strip any comment line that names a specific k_<name> (old decoy hint)
            t = re.sub(r'\n\s*//[^\n]*k_[a-z]+[^\n]*', '', t)
            p.write_text(t)

    manifest = {
        "trial": trial, "master_seed": master_seed,
        "key_names": names, "correct_kid": correct_kid, "decoy_kids": decoy_kids,
        "corpus_path": str(corpus_out), "corpus_size": CORPUS_SIZE,
        "reproduce": f"prepare_trial.py --trial {trial} --master-seed {master_seed}",
    }
    manifest_out.parent.mkdir(parents=True, exist_ok=True)
    manifest_out.write_text(json.dumps(manifest, indent=2))
    return correct_kid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--trial", type=int, required=True)
    ap.add_argument("--master-seed", type=int, required=True)
    ap.add_argument("--corpus-out", required=True)
    ap.add_argument("--manifest-out", required=True)
    ap.add_argument("--vestige-bin", default="$HOME/vestige/target/release/vestige")
    a = ap.parse_args()
    kid = prepare_trial(Path(a.repo), a.trial, a.master_seed, Path(a.corpus_out),
                        Path(a.manifest_out), a.vestige_bin)
    print(kid)  # last line = the trial's correct kid, for the caller to export


if __name__ == "__main__":
    main()
