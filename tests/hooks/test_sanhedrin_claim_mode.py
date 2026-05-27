import contextlib
import importlib.util
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[2]
HOOK_PATH = REPO_ROOT / "hooks" / "sanhedrin-local.py"


def load_sanhedrin():
    spec = importlib.util.spec_from_file_location("sanhedrin_local_under_test", HOOK_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@contextlib.contextmanager
def patched_attr(obj, name, value):
    sentinel = object()
    old = getattr(obj, name, sentinel)
    setattr(obj, name, value)
    try:
        yield
    finally:
        if old is sentinel:
            delattr(obj, name)
        else:
            setattr(obj, name, old)


class SanhedrinClaimModeTests(unittest.TestCase):
    def setUp(self):
        for key in (
            "VESTIGE_SANHEDRIN_CLAIM_MODE",
            "VESTIGE_SANHEDRIN_OUTPUT",
            "VESTIGE_SANHEDRIN_STAGE_FILE",
            "VESTIGE_SANHEDRIN_TRANSCRIPT",
            "VESTIGE_SANHEDRIN_ALLOW_COMMAND_LEDGER",
        ):
            os.environ.pop(key, None)
        self.sanhedrin = load_sanhedrin()
        self.sanhedrin.SANHEDRIN_ENDPOINT = "http://127.0.0.1:8080/v1/chat/completions"
        self.sanhedrin.MODEL = "test-verifier"

    @contextlib.contextmanager
    def isolated_receipt_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            state_dir = Path(tmp)
            core = self.sanhedrin.sanhedrin_core
            with patched_attr(core, "STATE_DIR", state_dir), patched_attr(
                core, "RECEIPTS_DIR", state_dir / "receipts"
            ), patched_attr(core, "LATEST_JSON", state_dir / "latest.json"), patched_attr(
                core, "LATEST_HTML", state_dir / "latest.html"
            ), patched_attr(
                core, "APPEALS_JSONL", state_dir / "appeals.jsonl"
            ), patched_attr(
                core, "COMMAND_RECEIPTS_JSONL", state_dir / "command-receipts.jsonl"
            ):
                yield state_dir

    def run_main(self, draft):
        stdin = io.StringIO(draft)
        stdout = io.StringIO()
        with mock.patch.object(sys, "stdin", stdin), mock.patch.object(sys, "stdout", stdout):
            self.sanhedrin.main()
        return stdout.getvalue().strip()

    def test_runtime_has_no_implicit_verifier_model_default(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            module = load_sanhedrin()

        self.assertEqual(module.SANHEDRIN_ENDPOINT, "")
        self.assertEqual(module.MODEL, "")

    def test_receipt_lock_blocks_unbacked_test_claim(self):
        with self.isolated_receipt_state() as state_dir:
            out = self.run_main("All tests passed.")

            self.assertIn("Receipt Lock", out)
            receipt = json.loads((state_dir / "latest.json").read_text(encoding="utf-8"))

        self.assertEqual(receipt["verdictBar"], "VETO")
        self.assertEqual(receipt["claims"][0]["decision"], "veto")
        self.assertEqual(receipt["claims"][0]["evidence_state"], "missing_receipt")

    def test_receipt_lock_allows_matching_success_receipt(self):
        with self.isolated_receipt_state() as state_dir, mock.patch.dict(
            os.environ, {"VESTIGE_SANHEDRIN_ALLOW_COMMAND_LEDGER": "1"}, clear=False
        ):
            (state_dir / "command-receipts.jsonl").write_text(
                json.dumps({
                    "command": "cargo test --workspace --release",
                    "exitCode": 0,
                    "success": True,
                }) + "\n",
                encoding="utf-8",
            )
            out = self.run_main("All tests passed.")
            receipt = json.loads((state_dir / "latest.json").read_text(encoding="utf-8"))

        self.assertEqual(out, "yes")
        self.assertNotEqual(receipt["verdictBar"], "VETO")
        self.assertEqual(receipt["claims"][0]["decision"], "pass")

    def test_receipt_lock_appeal_suppresses_same_fingerprint(self):
        with self.isolated_receipt_state() as state_dir:
            fingerprint = self.sanhedrin.sanhedrin_core.claim_fingerprint("All tests passed.")
            (state_dir / "appeals.jsonl").write_text(
                json.dumps({
                    "claimFingerprint": fingerprint,
                    "reason": "too_strict",
                    "status": "active",
                }) + "\n",
                encoding="utf-8",
            )
            out = self.run_main("All tests passed.")
            receipt = json.loads((state_dir / "latest.json").read_text(encoding="utf-8"))

        self.assertEqual(out, "yes")
        self.assertEqual(receipt["verdictBar"], "APPEALED")
        self.assertEqual(receipt["claims"][0]["decision"], "appealed")

    def test_receipt_lock_ignores_quotes_fences_and_hedged_verification(self):
        examples = [
            'The user said "all tests passed" earlier.',
            "> all tests passed\nI still need to verify this myself.",
            "```text\nall tests passed\n```",
            "I think the tests passed before, but let me verify.",
        ]
        for example in examples:
            with self.subTest(example=example), self.isolated_receipt_state() as state_dir:
                out = self.run_main(example)
                self.assertEqual(out, "yes")
                latest = state_dir / "latest.json"
                if latest.exists():
                    receipt = json.loads(latest.read_text(encoding="utf-8"))
                    self.assertNotEqual(receipt["verdictBar"], "VETO")

    def test_claim_mode_ignores_quoted_and_blockquoted_verification_text(self):
        examples = [
            'The user said "all tests passed" earlier.',
            "> all tests passed\nI still need to verify this myself.",
            "```text\nall tests passed\n```",
        ]
        env = {"VESTIGE_SANHEDRIN_CLAIM_MODE": "1", "VESTIGE_SANHEDRIN_OUTPUT": "json"}
        with mock.patch.dict(os.environ, env, clear=False), patched_attr(
            self.sanhedrin, "fetch_claim_evidence", lambda _claim: ([], True)
        ):
            for example in examples:
                with self.subTest(example=example):
                    out = self.run_main(example)
                    result = json.loads(out)

                    self.assertTrue(result["passed"], result)

    def test_receipt_lock_still_blocks_temporal_or_apostrophe_claims(self):
        examples = [
            "All tests passed before I pushed the fix.",
            "All tests passed earlier on the staging branch.",
            "All tests passed last run.",
            "Sam's tests passed today.",
        ]
        for example in examples:
            with self.subTest(example=example), self.isolated_receipt_state() as state_dir:
                out = self.run_main(example)
                receipt = json.loads((state_dir / "latest.json").read_text(encoding="utf-8"))

            self.assertIn("Receipt Lock", out)
            self.assertEqual(receipt["verdictBar"], "VETO")

    def test_loose_transcript_command_scan_is_disabled_by_default(self):
        with self.isolated_receipt_state() as state_dir:
            transcript = state_dir / "transcript.jsonl"
            transcript.write_text(
                json.dumps({
                    "role": "assistant",
                    "message": {
                        "content": 'I will not run it, but here is {"command":"cargo test","exit_code":0}.'
                    },
                }) + "\n",
                encoding="utf-8",
            )
            with mock.patch.dict(os.environ, {"VESTIGE_SANHEDRIN_TRANSCRIPT": str(transcript)}, clear=False):
                out = self.run_main("All tests passed.")
                receipt = json.loads((state_dir / "latest.json").read_text(encoding="utf-8"))

        self.assertIn("Receipt Lock", out)
        self.assertEqual(receipt["verdictBar"], "VETO")
        self.assertEqual(receipt["receipts"], [])

    def test_plain_sam_biographical_achievement_claim_is_check_worthy(self):
        claims = self.sanhedrin.extract_check_worthy_claims(
            "Sam graduated from Example University and won the Example AI Challenge."
        )

        self.assertGreaterEqual(len(claims), 1)
        self.assertTrue(any(claim.sam_critical for claim in claims))
        self.assertTrue(
            any(claim.claim_class in {"BIOGRAPHICAL", "ACHIEVEMENT"} for claim in claims)
        )
        self.assertTrue(any("Sam" in claim.text for claim in claims))

    def test_zero_high_trust_evidence_on_sam_critical_claim_blocks(self):
        def fail_if_judge_is_called(_claim, _evidence):
            self.fail("zero-evidence absence decisions should not require model judgment")

        env = {"VESTIGE_SANHEDRIN_CLAIM_MODE": "1", "VESTIGE_SANHEDRIN_OUTPUT": "json"}
        with mock.patch.dict(os.environ, env, clear=False), patched_attr(
            self.sanhedrin, "fetch_claim_evidence", lambda _claim: ([], True)
        ), patched_attr(self.sanhedrin, "judge_claim_with_model", fail_if_judge_is_called):
            out = self.run_main("Sam won first place at the Example AI Challenge.")

        result = json.loads(out)
        self.assertFalse(result["passed"])
        self.assertTrue(result["legacy_verdict"].startswith("no - "), result)
        self.assertEqual(result["verdicts"][0]["status"], "REFUTED_BY_ABSENCE")

    def test_missing_model_configuration_fails_open_except_receipt_lock(self):
        env = {
            "VESTIGE_SANHEDRIN_CLAIM_MODE": "1",
            "VESTIGE_SANHEDRIN_OUTPUT": "json",
        }
        with mock.patch.dict(os.environ, env, clear=False), patched_attr(
            self.sanhedrin, "SANHEDRIN_ENDPOINT", ""
        ), patched_attr(self.sanhedrin, "MODEL", ""), patched_attr(
            self.sanhedrin, "fetch_claim_evidence", lambda _claim: ([], True)
        ):
            out = self.run_main("Sam attended Example University.")

        result = json.loads(out)
        self.assertTrue(result["passed"], result)
        self.assertEqual(result["verdicts"][0]["status"], "NEI")
        self.assertIn("model not configured", result["verdicts"][0]["reason"])

    def test_vague_user_positive_claim_fails_closed(self):
        env = {"VESTIGE_SANHEDRIN_CLAIM_MODE": "1", "VESTIGE_SANHEDRIN_OUTPUT": "json"}
        with mock.patch.dict(os.environ, env, clear=False), patched_attr(
            self.sanhedrin, "fetch_claim_evidence", lambda _claim: ([], True)
        ):
            out = self.run_main("Sam won a few competitions and earned some prize money.")

        result = json.loads(out)
        self.assertFalse(result["passed"], result)
        self.assertEqual(result["verdicts"][0]["claim"]["claim_class"], "VAGUE-QUANTIFIER")
        self.assertEqual(result["verdicts"][0]["status"], "REFUTED_BY_ABSENCE")

    def test_retrieval_failure_on_sam_critical_claim_fails_open(self):
        def fail_if_judge_is_called(_claim, _evidence):
            self.fail("retrieval failures should fail open before model judgment")

        env = {"VESTIGE_SANHEDRIN_CLAIM_MODE": "1", "VESTIGE_SANHEDRIN_OUTPUT": "json"}
        with mock.patch.dict(os.environ, env, clear=False), patched_attr(
            self.sanhedrin, "fetch_claim_evidence", lambda _claim: ([], False)
        ), patched_attr(self.sanhedrin, "judge_claim_with_model", fail_if_judge_is_called):
            out = self.run_main("Sam won first place at the Example AI Challenge.")

        result = json.loads(out)
        self.assertTrue(result["passed"], result)
        self.assertEqual(result["legacy_verdict"], "yes")
        self.assertEqual(result["verdicts"][0]["status"], "NEI")
        self.assertIn("retrieval unavailable", result["verdicts"][0]["reason"])

    def test_current_turn_attribution_discourse_is_not_absence_blocked(self):
        env = {"VESTIGE_SANHEDRIN_CLAIM_MODE": "1", "VESTIGE_SANHEDRIN_OUTPUT": "json"}
        with mock.patch.dict(os.environ, env, clear=False), patched_attr(
            self.sanhedrin, "fetch_claim_evidence", lambda _claim: ([], True)
        ):
            out = self.run_main(
                "You asked me to audit the Sanhedrin hook, and I reviewed your requested changes."
            )

        result = json.loads(out)
        self.assertTrue(result["passed"], result)
        self.assertEqual(result["legacy_verdict"], "yes")
        self.assertEqual(result["claims_extracted"], 0)

    def test_discourse_framing_does_not_hide_embedded_sam_claim(self):
        examples = [
            "Per your request, Sam won first place at the Example AI Challenge.",
            "Sam won first place at the Example AI Challenge, which would be impressive.",
        ]
        env = {"VESTIGE_SANHEDRIN_CLAIM_MODE": "1", "VESTIGE_SANHEDRIN_OUTPUT": "json"}
        for example in examples:
            with self.subTest(example=example), mock.patch.dict(os.environ, env, clear=False), patched_attr(
                self.sanhedrin, "fetch_claim_evidence", lambda _claim: ([], True)
            ):
                out = self.run_main(example)

            result = json.loads(out)
            self.assertFalse(result["passed"], result)
            self.assertEqual(result["verdicts"][0]["status"], "REFUTED_BY_ABSENCE")
            self.assertIn("Sam won", result["verdicts"][0]["claim"]["text"])

    def test_leading_hypothetical_still_skips_embedded_claim(self):
        env = {"VESTIGE_SANHEDRIN_CLAIM_MODE": "1", "VESTIGE_SANHEDRIN_OUTPUT": "json"}
        with mock.patch.dict(os.environ, env, clear=False), patched_attr(
            self.sanhedrin, "fetch_claim_evidence", lambda _claim: ([], True)
        ):
            out = self.run_main("If Sam wins first place next time, he could claim the prize.")

        result = json.loads(out)
        self.assertTrue(result["passed"], result)
        self.assertEqual(result["claims_extracted"], 0)

    def test_subject_modal_prefix_skips_without_hiding_asserted_claim(self):
        env = {"VESTIGE_SANHEDRIN_CLAIM_MODE": "1", "VESTIGE_SANHEDRIN_OUTPUT": "json"}
        with mock.patch.dict(os.environ, env, clear=False), patched_attr(
            self.sanhedrin, "fetch_claim_evidence", lambda _claim: ([], True)
        ):
            nonassertive = self.run_main("Sam could win first place next time.")
            asserted = self.run_main(
                "Sam won first place at the Example AI Challenge and could collect prize money."
            )

        nonassertive_result = json.loads(nonassertive)
        asserted_result = json.loads(asserted)
        self.assertTrue(nonassertive_result["passed"], nonassertive_result)
        self.assertEqual(nonassertive_result["claims_extracted"], 0)
        self.assertFalse(asserted_result["passed"], asserted_result)
        self.assertEqual(asserted_result["verdicts"][0]["status"], "REFUTED_BY_ABSENCE")

    def test_malformed_deep_reference_response_fails_open(self):
        def fail_if_judge_is_called(_claim, _evidence):
            self.fail("malformed retrieval responses should fail open before model judgment")

        env = {"VESTIGE_SANHEDRIN_CLAIM_MODE": "1", "VESTIGE_SANHEDRIN_OUTPUT": "json"}
        for response in ({}, {"status": "error"}, {"errors": ["timeout"]}):
            with self.subTest(response=response):
                def fake_post_json(_url, _body, _timeout):
                    return response

                with mock.patch.dict(os.environ, env, clear=False), patched_attr(
                    self.sanhedrin, "post_json", fake_post_json
                ), patched_attr(self.sanhedrin, "judge_claim_with_model", fail_if_judge_is_called):
                    out = self.run_main("Sam won first place at the Example AI Challenge.")

                result = json.loads(out)
                self.assertTrue(result["passed"], result)
                self.assertEqual(result["verdicts"][0]["status"], "NEI")
                self.assertIn("retrieval unavailable", result["verdicts"][0]["reason"])

    def test_non_critical_technical_zero_evidence_does_not_block(self):
        def fail_if_judge_is_called(_claim, _evidence):
            self.fail("zero-evidence technical claims should fail open without model judgment")

        env = {"VESTIGE_SANHEDRIN_CLAIM_MODE": "1", "VESTIGE_SANHEDRIN_OUTPUT": "json"}
        with mock.patch.dict(os.environ, env, clear=False), patched_attr(
            self.sanhedrin, "fetch_claim_evidence", lambda _claim: ([], True)
        ), patched_attr(self.sanhedrin, "judge_claim_with_model", fail_if_judge_is_called):
            out = self.run_main(
                "Qwen3.6-35B can be served through an OpenAI-compatible chat endpoint."
            )

        result = json.loads(out)
        self.assertTrue(result["passed"])
        self.assertEqual(result["legacy_verdict"], "yes")
        self.assertEqual(result["verdicts"][0]["status"], "NEI")
        self.assertEqual(result["verdicts"][0]["claim"]["claim_class"], "TECHNICAL")

    def test_claim_sampling_keeps_late_high_severity_claim(self):
        technical = " ".join(
            f"The /tmp/example_{i}.py script calls the MCP endpoint successfully."
            for i in range(12)
        )
        claims = self.sanhedrin.extract_check_worthy_claims(
            f"{technical} Sam won first place at the Example AI Challenge."
        )

        self.assertLessEqual(len(claims), self.sanhedrin.MAX_CLAIMS)
        self.assertTrue(
            any(
                claim.sam_critical and claim.claim_class == "ACHIEVEMENT"
                for claim in claims
            ),
            claims,
        )

    def test_fetch_evidence_truncates_on_python_character_boundary(self):
        emoji_out = self.sanhedrin.truncate_chars(("a" * 4) + "🙂" + "tail", 8)
        combining_out = self.sanhedrin.truncate_chars("Cafe\u0301 tail", 8)

        self.assertEqual(emoji_out, "aaaa🙂...")
        self.assertEqual(combining_out, "Cafe...")
        self.assertNotIn("\ufffd", emoji_out + combining_out)
        self.assertFalse(self.sanhedrin.unicodedata.combining(combining_out[-4]))
        (emoji_out + combining_out).encode("utf-8")

    def test_staged_evidence_is_used_without_smart_ingest_or_durable_write(self):
        with tempfile.TemporaryDirectory() as tmp:
            staged_path = Path(tmp) / "sanhedrin-staged-evidence.json"
            staged = [
                {
                    "id": "samstage2",
                    "role": "memory",
                    "trust": 0.89,
                    "preview": "Sam's final result was second place with no payout.",
                }
            ]
            staged_path.write_text(json.dumps(staged), encoding="utf-8")

            post_urls = []

            def fake_post_json(url, body, _timeout):
                post_urls.append(url)
                if "smart_ingest" in url or "/api/memories" in url:
                    self.fail(f"staged evidence path attempted durable write to {url}: {body}")
                self.assertEqual(url, "http://127.0.0.1:3927/api/deep_reference")
                return {"confidence": 0.0, "evidence": []}

            env = {
                "VESTIGE_SANHEDRIN_CLAIM_MODE": "1",
                "VESTIGE_SANHEDRIN_OUTPUT": "json",
                "VESTIGE_SANHEDRIN_STAGE_FILE": str(staged_path),
            }
            with mock.patch.dict(os.environ, env, clear=False), patched_attr(
                self.sanhedrin, "post_json", fake_post_json
            ), patched_attr(self.sanhedrin, "VESTIGE_ENDPOINT", "http://127.0.0.1:3927/api/deep_reference"):
                out = self.run_main("Sam won first place and earned prize money.")

        result = json.loads(out)
        verdict = result["verdicts"][0]
        self.assertFalse(result["passed"], result)
        self.assertEqual(result["staged_evidence_count"], 1)
        self.assertEqual(verdict["status"], "REFUTED_BY_ABSENCE")
        self.assertEqual(verdict["durable_evidence_count"], 0)
        self.assertEqual(verdict["high_trust_evidence_count"], 1)
        self.assertEqual(post_urls, ["http://127.0.0.1:3927/api/deep_reference"])

    def test_staged_only_refuted_verdict_is_downgraded_without_durable_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            staged_path = Path(tmp) / "sanhedrin-staged-evidence.json"
            staged_path.write_text(
                json.dumps(
                    [
                        {
                            "id": "stage-tech",
                            "trust": 0.95,
                            "preview": "Qwen3.6-35B cannot be served through a chat endpoint.",
                        }
                    ]
                ),
                encoding="utf-8",
            )

            def fake_post_json(url, _body, _timeout):
                if url == self.sanhedrin.VESTIGE_ENDPOINT:
                    return {"confidence": 0.0, "evidence": []}
                if url == self.sanhedrin.SANHEDRIN_ENDPOINT:
                    return {
                        "choices": [
                            {
                                "message": {
                                    "content": json.dumps(
                                        {
                                            "status": "REFUTED",
                                            "class": "TECHNICAL",
                                            "reason": "Staged evidence contradicts the claim.",
                                            "evidence_ids": ["stage-tech"],
                                        }
                                    )
                                }
                            }
                        ]
                    }
                self.fail(f"unexpected post_json URL: {url}")

            env = {
                "VESTIGE_SANHEDRIN_CLAIM_MODE": "1",
                "VESTIGE_SANHEDRIN_OUTPUT": "json",
                "VESTIGE_SANHEDRIN_STAGE_FILE": str(staged_path),
            }
            with mock.patch.dict(os.environ, env, clear=False), patched_attr(
                self.sanhedrin, "post_json", fake_post_json
            ), patched_attr(
                self.sanhedrin, "SANHEDRIN_ENDPOINT", "http://127.0.0.1:8080/v1/chat/completions"
            ), patched_attr(self.sanhedrin, "MODEL", "test-verifier"):
                out = self.run_main(
                    "Qwen3.6-35B can be served through an OpenAI-compatible chat endpoint."
                )

        result = json.loads(out)
        verdict = result["verdicts"][0]
        self.assertTrue(result["passed"], result)
        self.assertEqual(verdict["status"], "NEI")
        self.assertEqual(verdict["durable_evidence_count"], 0)
        self.assertIn("Durable evidence required", verdict["reason"])

    def test_staged_only_supported_verdict_is_downgraded_without_durable_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            staged_path = Path(tmp) / "sanhedrin-staged-evidence.json"
            staged_path.write_text(
                json.dumps(
                    [
                        {
                            "id": "stage-tech",
                            "trust": 0.95,
                            "preview": "Qwen3.6-35B can be served through a chat endpoint.",
                        }
                    ]
                ),
                encoding="utf-8",
            )

            def fake_post_json(url, _body, _timeout):
                if url == self.sanhedrin.VESTIGE_ENDPOINT:
                    return {"confidence": 0.0, "evidence": []}
                if url == self.sanhedrin.SANHEDRIN_ENDPOINT:
                    return {
                        "choices": [
                            {
                                "message": {
                                    "content": json.dumps(
                                        {
                                            "status": "SUPPORTED",
                                            "class": "TECHNICAL",
                                            "reason": "Staged evidence supports the claim.",
                                            "evidence_ids": ["stage-tech"],
                                        }
                                    )
                                }
                            }
                        ]
                    }
                self.fail(f"unexpected post_json URL: {url}")

            env = {
                "VESTIGE_SANHEDRIN_CLAIM_MODE": "1",
                "VESTIGE_SANHEDRIN_OUTPUT": "json",
                "VESTIGE_SANHEDRIN_STAGE_FILE": str(staged_path),
            }
            with mock.patch.dict(os.environ, env, clear=False), patched_attr(
                self.sanhedrin, "post_json", fake_post_json
            ), patched_attr(
                self.sanhedrin, "SANHEDRIN_ENDPOINT", "http://127.0.0.1:8080/v1/chat/completions"
            ), patched_attr(self.sanhedrin, "MODEL", "test-verifier"):
                out = self.run_main(
                    "Qwen3.6-35B can be served through an OpenAI-compatible chat endpoint."
                )

        result = json.loads(out)
        verdict = result["verdicts"][0]
        self.assertTrue(result["passed"], result)
        self.assertEqual(verdict["status"], "NEI")
        self.assertEqual(verdict["durable_evidence_count"], 0)
        self.assertIn("Durable evidence required", verdict["reason"])

    def test_supported_verdict_with_durable_evidence_is_preserved(self):
        evidence = [
            self.sanhedrin.EvidenceItem(
                id="mem-durable",
                preview="A reliable memory says this backend can use a compatible endpoint.",
                trust=0.95,
                durable=True,
                source="vestige",
            )
        ]
        claim = self.sanhedrin.Claim(
            text="Qwen3.6-35B can be served through an OpenAI-compatible chat endpoint.",
            claim_class="TECHNICAL",
            source_index=0,
            sam_critical=False,
        )
        verdict = self.sanhedrin.validate_structured_verdict(
            claim,
            {"status": "SUPPORTED", "class": "TECHNICAL", "reason": "Evidence supports it."},
            evidence,
        )

        self.assertEqual(verdict.status, "SUPPORTED")

    def test_openai_key_is_not_forwarded_to_arbitrary_or_vestige_endpoints(self):
        captured_headers = []

        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def read(self):
                return b"{}"

        def fake_urlopen(req, timeout=None):
            captured_headers.append(dict(req.header_items()))
            return FakeResponse()

        env = {"OPENAI_API_KEY": "real-openai-key"}
        with mock.patch.dict(os.environ, env, clear=False), patched_attr(
            self.sanhedrin, "SANHEDRIN_ENDPOINT", "http://127.0.0.1:8080/v1/chat/completions"
        ), mock.patch.object(
            self.sanhedrin.urllib.request, "urlopen", fake_urlopen
        ):
            self.sanhedrin.post_json(self.sanhedrin.SANHEDRIN_ENDPOINT, {}, 1)
            self.sanhedrin.post_json(self.sanhedrin.VESTIGE_ENDPOINT, {}, 1)

        self.assertTrue(captured_headers)
        self.assertTrue(all("Authorization" not in headers for headers in captured_headers))

    def test_sanhedrin_api_key_only_goes_to_configured_sanhedrin_endpoint(self):
        captured_headers = []

        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def read(self):
                return b"{}"

        def fake_urlopen(req, timeout=None):
            captured_headers.append(dict(req.header_items()))
            return FakeResponse()

        env = {"VESTIGE_SANHEDRIN_API_KEY": "sanhedrin-only-key"}
        with mock.patch.dict(os.environ, env, clear=False), patched_attr(
            self.sanhedrin, "SANHEDRIN_ENDPOINT", "http://127.0.0.1:8080/v1/chat/completions"
        ), mock.patch.object(
            self.sanhedrin.urllib.request, "urlopen", fake_urlopen
        ):
            self.sanhedrin.post_json(self.sanhedrin.SANHEDRIN_ENDPOINT, {}, 1)
            self.sanhedrin.post_json(self.sanhedrin.VESTIGE_ENDPOINT, {}, 1)

        self.assertIn("Authorization", captured_headers[0])
        self.assertNotIn("Authorization", captured_headers[1])

    def test_strict_openai_body_omits_backend_specific_fields(self):
        with patched_attr(self.sanhedrin, "SANHEDRIN_BACKEND", "openai"):
            body = self.sanhedrin.sanhedrin_body(
                [{"role": "user", "content": "judge"}],
                128,
            )

        self.assertNotIn("top_k", body)
        self.assertNotIn("seed", body)
        self.assertNotIn("chat_template_kwargs", body)

    def test_mlx_body_keeps_backend_specific_fields(self):
        with patched_attr(self.sanhedrin, "SANHEDRIN_BACKEND", "mlx"):
            body = self.sanhedrin.sanhedrin_body(
                [{"role": "user", "content": "judge"}],
                128,
            )

        self.assertEqual(body["top_k"], 1)
        self.assertEqual(body["chat_template_kwargs"], {"enable_thinking": False})

    def test_staged_only_legacy_refuted_line_is_downgraded_without_durable_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            staged_path = Path(tmp) / "sanhedrin-staged-evidence.json"
            staged_path.write_text(
                json.dumps(
                    [
                        {
                            "id": "stage-tech",
                            "trust": 0.95,
                            "preview": "Qwen3.6-35B cannot be served through a chat endpoint.",
                        }
                    ]
                ),
                encoding="utf-8",
            )

            def fake_post_json(url, _body, _timeout):
                if url == self.sanhedrin.VESTIGE_ENDPOINT:
                    return {"confidence": 0.0, "evidence": []}
                if url == self.sanhedrin.SANHEDRIN_ENDPOINT:
                    return {
                        "choices": [
                            {
                                "message": {
                                    "content": (
                                        "no - [Sanhedrin Veto] [TECHNICAL]: "
                                        "Staged evidence contradicts the claim."
                                    )
                                }
                            }
                        ]
                    }
                self.fail(f"unexpected post_json URL: {url}")

            env = {
                "VESTIGE_SANHEDRIN_CLAIM_MODE": "1",
                "VESTIGE_SANHEDRIN_OUTPUT": "json",
                "VESTIGE_SANHEDRIN_STAGE_FILE": str(staged_path),
            }
            with mock.patch.dict(os.environ, env, clear=False), patched_attr(
                self.sanhedrin, "post_json", fake_post_json
            ), patched_attr(
                self.sanhedrin, "SANHEDRIN_ENDPOINT", "http://127.0.0.1:8080/v1/chat/completions"
            ), patched_attr(self.sanhedrin, "MODEL", "test-verifier"):
                out = self.run_main(
                    "Qwen3.6-35B can be served through an OpenAI-compatible chat endpoint."
                )

        result = json.loads(out)
        verdict = result["verdicts"][0]
        self.assertTrue(result["passed"], result)
        self.assertEqual(verdict["status"], "NEI")
        self.assertEqual(verdict["durable_evidence_count"], 0)
        self.assertIn("Durable evidence required", verdict["reason"])

    def test_current_turn_discourse_patterns_are_not_claims(self):
        examples = [
            "You asked for maximum subagents, so I audited the hook.",
            "Your request was to verify the installer env preservation.",
            "Per your request, I reviewed the Sanhedrin stop hook.",
            "Sam asked me to go all in on the Sanhedrin patch.",
            "The user requested maximum subagents for this implementation.",
        ]
        for example in examples:
            with self.subTest(example=example):
                self.assertEqual(self.sanhedrin.extract_check_worthy_claims(example), [])


if __name__ == "__main__":
    unittest.main()
