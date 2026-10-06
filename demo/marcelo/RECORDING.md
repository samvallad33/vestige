# Recording this demo

Vestige is a cognitive, deterministic memory-transaction security OS for AI agents.

Run this on your Mac, in a terminal, from a checkout of this branch. Setup happens before you start recording. The two commands below are what you type on camera, in this order.

## Check the Mac first

Xcode command line tools:

```sh
xcode-select -p
```

That prints a path such as `/Library/Developer/CommandLineTools`. If it fails, install them:

```sh
xcode-select --install
```

Rust, via rustup:

```sh
rustc --version
cargo --version
```

Both commands should print a version. If either is missing, install from https://rustup.rs and open a new terminal.

Python:

```sh
python3 --version
python3 -m venv "$HOME/vestige-venv-check"
rm -rf "$HOME/vestige-venv-check"
```

The venv command should finish quietly. Use a python.org or Homebrew Python if the system one cannot create a virtual environment.

Git:

```sh
git --version
```

## Setup, once, before you record

From the repository root:

```sh
bash demo/marcelo/setup.sh
```

This clones Vestige, checks out commit `b4bcd52b81`, builds the default-feature binaries, fetches the uv checkout, and prepares the baseline's Python environment, including its model download. The default folder is `$HOME/vestige-demo`. Override it with `DEMO_HOME` if you want the files somewhere else. Use the same `DEMO_HOME` for every command.

On the Linux machine that produced the pasted output below, this took 3 minutes 31 seconds. The Rust crates were already in the local cargo cache, so a Mac downloading them for the first time takes longer. Wait until the last line is:

```text
ready
```

## While you record

From the repository root, type these two commands, in this order.

```sh
DEMO_PAUSE=1 bash demo/marcelo/vestige-side.sh
```

```sh
bash demo/marcelo/rag-side.sh
```

The second command reads the commit list and the failure text written by the first. There is a pause of about 2 seconds after each plain-English line on the Vestige side.

## What you are looking at

The first lines say this is pre-release code from PR #445, commit `b4bcd52b81`, not the v4.1.1 release.

"This step creates a fresh empty data directory" starts this run from an empty store. The path is under your demo folder.

"This step records uv history through the parent of b52d489" reads the same repository history as the regression test, stopping at the parent of that commit. The page length of 20 is the limit in that test. On this page the cause is the third commit. `b52d489` is the revert, so it stays outside the record. The lines under it should show 20 ingested commits, `351d602` ingested, `d2f58d9` ingested, and `b52d489` not ingested.

"This step saves one failure record" writes the failure text from the test and links it with `derived_from` to the revision where it was seen, `351d602`.

"This step walks backward from that failure record" ranks the commits that walk reaches. On the run below, `d2f58d9` is rank 1. `causal_walk seconds` is measured on that run, not filled in ahead of time.

"This step replays the signed log" rebuilds the log and prints both digests. `MATCH` means they are the same.

"This step checks the store with strata-verify" checks the store and prints the signing-key fingerprint, then `OK`. A new data directory creates its own signing key, so the fingerprint and the two digests on your machine will differ from the paste. The rank, the ingested count, and `MATCH` / `OK` are the lines to read.

The second command is the baseline. It ranks those same 20 commits from the same failure text and prints its own top 9, then the rank of `d2f58d9`. On the run below that rank is 20. The corpus size on screen should be 20, the same count as the Vestige side.

## Output from the fair-cut run

Vestige side:

```text
This is pre-release code from PR #445 (commit b4bcd52b81), not the v4.1.1 release.
HEAD b4bcd52b81402722d6c99b96f461278a171d110d

This step creates a fresh empty data directory for this run.
data directory: /tmp/vestige-demo/data

This step records uv history through the parent of b52d489, so the later revert stays outside the record.
ingest round 1: created=20 remaining=0 stopped=false
history rev: d3f06de4f5fca1a9bb43f11b8e469fce306e94db
ingested commits: 20
351d602d86c484a39bc537f1eb99866ea2c25fc1 ingested
d2f58d92991fa08b24596fcc6c6472dc5015d3bc ingested
b52d48973fe9ddb2e78b663ec48a1a68f7e7802d not ingested

This step saves one failure record, labeled as the failure, tied to the revision where it was seen.
observed revision 351d602d86c484a39bc537f1eb99866ea2c25fc1
revision record mem-0000000000000049
failure record mem-0000000000000be9
label tag: failure
failure: uv publish raises `error decoding response body` on 0.5.12. 0.5.11 works. CI https://github.com/andrew000/FTL-Extract/actions/runs/12509313775/job/34898612613#step:8:12
link: derived_from 351d602d86c484a39bc537f1eb99866ea2c25fc1

This step walks backward from that failure record and ranks the commits it reaches.
#1 d2f58d92991fa08b24596fcc6c6472dc5015d3bc edges=touched
#2 b6697a777c301f68ec4f8a81f1707badf223e837 edges=touched
#3 74112553bf3c1e675dddc5ce25c3f7d3b6f49656 edges=touched,derived_from
#4 79dce7391e17c9872a96e3d2cfe186c3a94ee1e0 edges=touched,derived_from
#5 0b5c0220b5a563ec67c1d4d407e636e0c8d17291 edges=touched,derived_from
#6 e6126ce0dc105c329dac4d45cf74c2ea8c944882 edges=touched,derived_from
#7 20df970a567f3a73448244b68dd5517ab738eaeb edges=touched,derived_from
#8 f40da39bafd375a549df4324d302d3257481c3c7 edges=touched,derived_from
#9 bec8468183c7cc1697ad1d34a5eb6087ec5c8a90 edges=touched,derived_from
causal_walk seconds: 0.025558
cause d2f58d92991fa08b24596fcc6c6472dc5015d3bc is rank 1

This step replays the signed log and rebuilds it, then compares the two digests.
live digest: 801b523d903b6180e75a6f64acb26f2f84444c5fa309011a65c47e60e8d96406
replayed digest: 801b523d903b6180e75a6f64acb26f2f84444c5fa309011a65c47e60e8d96406
MATCH

This step checks the store with strata-verify and prints the signing-key fingerprint.
{
  "failures": [],
  "frames_total": 3056,
  "key_fingerprint": "997783c46c66c60c953ffb87e4c19d21f9f8ae7c63629ad9030eb77ba9311409",
  "key_pin": "strata.key",
  "key_pin_note": "no migration receipt. Folder pin is strata.key in the log directory. A trailer signature must match it. No receipt-signing.key was required.",
  "ok": true,
  "segments": 1
}
key fingerprint: 997783c46c66c60c953ffb87e4c19d21f9f8ae7c63629ad9030eb77ba9311409
OK
```

Baseline:

```text
Standard vector search baseline (not Vestige)
model: all-MiniLM-L6-v2
corpus commits: 20
query:
failure: uv publish raises `error decoding response body` on 0.5.12. 0.5.11 works. CI https://github.com/andrew000/FTL-Extract/actions/runs/12509313775/job/34898612613#step:8:12
top 9:
#1 3435777e87b8620cfdbac7b1a79a6f4abaac70da cosine=0.416965
#2 3cb723220e516edae57930206cf3faffaa1b8e5b cosine=0.397659
#3 0b5c0220b5a563ec67c1d4d407e636e0c8d17291 cosine=0.379415
#4 79dce7391e17c9872a96e3d2cfe186c3a94ee1e0 cosine=0.371084
#5 bec8468183c7cc1697ad1d34a5eb6087ec5c8a90 cosine=0.361305
#6 7c47a457d981042f9c871cd4a1dd1307150178c9 cosine=0.347425
#7 ddde9481e3c4fd3fb1d09d3e8a54e8209eedf463 cosine=0.333427
#8 d3f06de4f5fca1a9bb43f11b8e469fce306e94db cosine=0.332630
#9 351d602d86c484a39bc537f1eb99866ea2c25fc1 cosine=0.320223
cause d2f58d92991fa08b24596fcc6c6472dc5015d3bc rank: 20
```

## Longer history

Leave `DEMO_HISTORY` unset for the 20-commit page above.

To record 300 commits that still end at that same parent, run setup with the variable set. That fetches 400 commits, the history length plus 100, so the page has its parents:

```sh
DEMO_HISTORY=300 bash demo/marcelo/setup.sh
```

Then pass the same variable to both live commands:

```sh
DEMO_HISTORY=300 DEMO_PAUSE=1 bash demo/marcelo/vestige-side.sh
DEMO_HISTORY=300 bash demo/marcelo/rag-side.sh
```

On the machine that produced the paste below, the Vestige side took 496.708 seconds. The first ingest round wrote all 300 commits. The budget flag stayed set through round 8, and round 9 reported the page finished. The walk listed 10 commits, and `d2f58d9` was rank 1. The baseline corpus was 300 commits, and that same commit was rank 266.

Vestige side:

```text
This is pre-release code from PR #445 (commit b4bcd52b81), not the v4.1.1 release.
HEAD b4bcd52b81402722d6c99b96f461278a171d110d

This step creates a fresh empty data directory for this run.
data directory: /tmp/vestige-demo/data

This step records uv history through the parent of b52d489, so the later revert stays outside the record.
ingest round 1: created=300 remaining=0 stopped=true
ingest round 2: created=0 remaining=0 stopped=true
ingest round 3: created=0 remaining=0 stopped=true
ingest round 4: created=0 remaining=0 stopped=true
ingest round 5: created=0 remaining=0 stopped=true
ingest round 6: created=0 remaining=0 stopped=true
ingest round 7: created=0 remaining=0 stopped=true
ingest round 8: created=0 remaining=0 stopped=true
ingest round 9: created=0 remaining=0 stopped=false
history rev: d3f06de4f5fca1a9bb43f11b8e469fce306e94db
ingested commits: 300
351d602d86c484a39bc537f1eb99866ea2c25fc1 ingested
d2f58d92991fa08b24596fcc6c6472dc5015d3bc ingested
b52d48973fe9ddb2e78b663ec48a1a68f7e7802d not ingested

This step saves one failure record, labeled as the failure, tied to the revision where it was seen.
observed revision 351d602d86c484a39bc537f1eb99866ea2c25fc1
revision record mem-00000000000004c5
failure record mem-0000000000008069
label tag: failure
failure: uv publish raises `error decoding response body` on 0.5.12. 0.5.11 works. CI https://github.com/andrew000/FTL-Extract/actions/runs/12509313775/job/34898612613#step:8:12
link: derived_from 351d602d86c484a39bc537f1eb99866ea2c25fc1

This step walks backward from that failure record and ranks the commits it reaches.
#1 d2f58d92991fa08b24596fcc6c6472dc5015d3bc edges=touched
#2 b6697a777c301f68ec4f8a81f1707badf223e837 edges=touched
#3 cf20673197f2073aa2ccde999423b816d3cb5ca5 edges=corrects
#4 74112553bf3c1e675dddc5ce25c3f7d3b6f49656 edges=touched,derived_from
#5 79dce7391e17c9872a96e3d2cfe186c3a94ee1e0 edges=touched,derived_from
#6 0b5c0220b5a563ec67c1d4d407e636e0c8d17291 edges=touched,derived_from
#7 e6126ce0dc105c329dac4d45cf74c2ea8c944882 edges=touched,derived_from
#8 20df970a567f3a73448244b68dd5517ab738eaeb edges=touched,derived_from
#9 f40da39bafd375a549df4324d302d3257481c3c7 edges=touched,derived_from
#10 bec8468183c7cc1697ad1d34a5eb6087ec5c8a90 edges=touched,derived_from
causal_walk seconds: 0.047502
cause d2f58d92991fa08b24596fcc6c6472dc5015d3bc is rank 1

This step replays the signed log and rebuilds it, then compares the two digests.
live digest: b0faa8aead5b5fe620fa8829b7920fae7968468f91432e552d7bbde1b683a7b6
replayed digest: b0faa8aead5b5fe620fa8829b7920fae7968468f91432e552d7bbde1b683a7b6
MATCH

This step checks the store with strata-verify and prints the signing-key fingerprint.
{
  "failures": [],
  "frames_total": 32880,
  "key_fingerprint": "de6f35cf26c94d8e241132fede3495a8b7eba8c7fa7483571c5a2ae02690bdb8",
  "key_pin": "strata.key",
  "key_pin_note": "no migration receipt. Folder pin is strata.key in the log directory. A trailer signature must match it. No receipt-signing.key was required.",
  "ok": true,
  "segments": 1
}
key fingerprint: de6f35cf26c94d8e241132fede3495a8b7eba8c7fa7483571c5a2ae02690bdb8
OK
```

Baseline:

```text
Standard vector search baseline (not Vestige)
model: all-MiniLM-L6-v2
corpus commits: 300
query:
failure: uv publish raises `error decoding response body` on 0.5.12. 0.5.11 works. CI https://github.com/andrew000/FTL-Extract/actions/runs/12509313775/job/34898612613#step:8:12
top 9:
#1 25045cb3e839bc4b661034065134190b57786239 cosine=0.500619
#2 33cb3497aa682042b53753f498b9fa93c3bcfac4 cosine=0.476336
#3 d4c2c46f6e6829293ca36749dfda869d52463d9c cosine=0.475592
#4 d257bea72059a5fa6c4c1e57298db4b95a77f256 cosine=0.473588
#5 6cc7a560f72a9e34907df5309d61980dac52a044 cosine=0.465514
#6 1dc0276458795d314375d62272e7a3887c261608 cosine=0.459151
#7 535ab69508d815b62f43f379cd49cb5be637cdda cosine=0.458827
#8 321101d340d7966e02b10b20f94704de0621e3d1 cosine=0.457959
#9 cb3fefff15f7795cd1877d82ea821b0a4896b6dc cosine=0.457928
cause d2f58d92991fa08b24596fcc6c6472dc5015d3bc rank: 266
```

## Do not say

- No case counts.
- No "11ms".
- No "40-60%".
- No "whitepaper".
- Do not call this comparison an answer to replay-ordering comparisons.
- Do not describe the Vestige side in similarity terms. Vestige is a cognitive, deterministic memory-transaction security OS for AI agents.
