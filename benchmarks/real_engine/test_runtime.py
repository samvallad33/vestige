"""Infrastructure invariants; no provider or local-model calls are made."""
from concurrent.futures import ProcessPoolExecutor
import asyncio
import hashlib
import json
from pathlib import Path
import tempfile
import time
import unittest

try:
    from .runtime import (AsyncAPIPool, BudgetExceeded, BudgetLedger, Checkpoint,
                          ReservationExists, append_jsonl, read_jsonl, record_key,
                          shard_id, status_loop, strict_merge, write_shard_map)
except ImportError:
    from runtime import (AsyncAPIPool, BudgetExceeded, BudgetLedger, Checkpoint,
                         ReservationExists, append_jsonl, read_jsonl, record_key,
                         shard_id, status_loop, strict_merge, write_shard_map)


def race_reserve(args):
    path, index = args
    ledger = BudgetLedger(path)
    try:
        ledger.reserve(f"call-{index}", 2)
        return True
    except BudgetExceeded:
        return False


class LedgerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "budget.db"

    def tearDown(self):
        self.tmp.cleanup()

    def test_reservations_are_process_safe(self):
        BudgetLedger(self.path)
        with ProcessPoolExecutor(max_workers=8) as pool:
            accepted = list(pool.map(race_reserve, [(self.path, i) for i in range(50)]))
        self.assertEqual(sum(accepted), 12)
        self.assertEqual(BudgetLedger(self.path).snapshot()["committed"], 24)

    def test_settlement_and_restart_do_not_reuse_attempt_id(self):
        ledger = BudgetLedger(self.path)
        ledger.reserve("id", 4)
        ledger.settle("id", 1.25, {"input_tokens": 5, "output_tokens": 2})
        ledger = BudgetLedger(self.path)
        self.assertEqual(ledger.snapshot()["spent"], 1.25)
        self.assertEqual(ledger.snapshot()["remaining"], 23.75)
        with self.assertRaises(ReservationExists):
            ledger.reserve("id", 4)
        ledger.settle("id", 1.25, {"input_tokens": 5, "output_tokens": 2})
        with self.assertRaises(ValueError):
            ledger.settle("id", 1.20, {})

    def test_unknown_failure_keeps_its_reservation(self):
        ledger = BudgetLedger(self.path)
        ledger.reserve("unknown", 25)
        with self.assertRaises(BudgetExceeded):
            BudgetLedger(self.path).reserve("next", 0.001)
        with self.assertRaises(ValueError):
            ledger.confirm_unbilled("unknown", {})
        ledger.confirm_unbilled("unknown", {"provider_billed_tokens": 0})
        self.assertEqual(ledger.snapshot()["remaining"], 25)

    def test_invalid_cost_and_changed_limit_rejected(self):
        ledger = BudgetLedger(self.path)
        for amount in (0, -1, "NaN", "Infinity"):
            with self.assertRaises(ValueError):
                ledger.reserve("bad", amount)
        with self.assertRaises(ValueError):
            BudgetLedger(self.path, 24)

    def test_upper_bound_violation_records_charge_and_stops_every_reserve(self):
        ledger = BudgetLedger(self.path)
        ledger.reserve("underestimated", 1)
        with self.assertRaises(BudgetExceeded):
            ledger.settle("underestimated", 1.01, {"input_tokens": 8})
        self.assertEqual(ledger.snapshot()["spent"], 1.01)
        with self.assertRaises(BudgetExceeded):
            ledger.reserve("next", 1)


class CheckpointTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.identity = {"dataset_sha256": "data", "binary_sha256": "engine", "model": "nano", "prompt": "p"}

    def tearDown(self):
        self.tmp.cleanup()

    def test_restart_skips_only_complete_same_identity_items(self):
        path = self.root / "worker-0.jsonl"
        checkpoint = Checkpoint(path, self.identity)
        row = {"item_id": "a", "trial": 0, "response": "hello"}
        checkpoint.append_failure({"item_id": "b", "trial": 0, "error": "timeout"})
        checkpoint.append(row)
        restored = Checkpoint(path, self.identity)
        self.assertTrue(restored.contains(row))
        self.assertFalse(restored.contains({"item_id": "b", "trial": 0}))
        self.assertEqual(len(read_jsonl(path.with_suffix(".errors.jsonl"))), 1)
        with self.assertRaises(ValueError):
            restored.append(row)
        with self.assertRaises(ValueError):
            Checkpoint(path, {**self.identity, "prompt": "changed"})

    def test_merge_rejects_missing_duplicates_and_unexpected(self):
        paths = [self.root / "worker-0.jsonl", self.root / "worker-1.jsonl"]
        expected = [record_key({"item_id": i, "trial": 0}) for i in ("a", "b")]
        append_jsonl(paths[0], {"item_id": "a", "trial": 0})
        output = self.root / "merged.jsonl"
        with self.assertRaisesRegex(ValueError, "missing=1"):
            strict_merge(paths, expected, output)
        self.assertFalse(output.exists())
        append_jsonl(paths[1], {"item_id": "b", "trial": 0})
        self.assertEqual(len(strict_merge(paths, expected, output)), 2)
        append_jsonl(paths[1], {"item_id": "a", "trial": 0})
        with self.assertRaisesRegex(ValueError, "duplicate"):
            strict_merge(paths, expected, output)
        with self.assertRaisesRegex(ValueError, "unexpected=1"):
            strict_merge([paths[0]], [], output)

    def test_trials_remain_separate(self):
        rows = [{"item_id": "a", "trial": i} for i in range(3)]
        path = self.root / "worker.jsonl"
        for row in rows:
            append_jsonl(path, row)
        self.assertEqual(len(strict_merge([path], map(record_key, rows), self.root / "merged")), 3)

    def test_corrupt_checkpoint_is_not_silently_dropped(self):
        path = self.root / "worker.jsonl"
        path.write_text('{"item_id":', encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "corrupt checkpoint"):
            Checkpoint(path, self.identity)

    def test_stable_hash_routing(self):
        ids = ["alpha", "beta", "gamma"]
        manifest = write_shard_map(self.root / "shards.json", ids, seed=123)
        self.assertEqual(manifest["seed"], 123)
        for item in ids:
            expected = int(hashlib.sha256(item.encode()).hexdigest(), 16) % 10
            self.assertEqual(shard_id(item), expected)
        reordered = write_shard_map(self.root / "other.json", reversed(ids), seed=123)
        self.assertEqual(manifest, reordered)


class HTTPFailure(Exception):
    def __init__(self, code):
        self.status_code = code


class PoolTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.ledger = BudgetLedger(self.root / "budget.db")

    async def asyncTearDown(self):
        self.tmp.cleanup()

    async def test_global_limit_across_pool_instances(self):
        active = peak = 0
        pools = [AsyncAPIPool(self.ledger) for _ in range(3)]
        async def send():
            nonlocal active, peak
            active += 1
            peak = max(active, peak)
            await asyncio.sleep(0.005)
            active -= 1
            return {"usage": {"input_tokens": 1}}
        await asyncio.gather(*(pools[i % 3].request(str(i), .001, send,
                                                   lambda r: (.0005, r["usage"])) for i in range(100)))
        self.assertEqual(peak, 40)
        self.assertEqual(self.ledger.snapshot()["spent"], .05)

    async def test_connect_timeout_retries_and_retains_ambiguous_reservations(self):
        import httpx
        attempts=0
        async def send():
            nonlocal attempts
            attempts+=1
            if attempts<3:
                raise httpx.ConnectTimeout('timeout')
            return {'usage':{'prompt_tokens':1,'completion_tokens':1}}
        async def sleep(delay):
            pass
        pool=AsyncAPIPool(self.ledger,sleep=sleep)
        await pool.request('timeout-retry',.01,send,lambda r:(.001,r['usage']))
        self.assertEqual(attempts,3)
        self.assertEqual(self.ledger.snapshot()['reserved'],.02)
        self.assertEqual(self.ledger.snapshot()['spent'],.001)

    async def test_six_retries_with_jitter_and_all_failed_attempts_reserved(self):
        delays, events = [], []
        async def sleep(delay):
            delays.append(delay)
        async def send():
            raise HTTPFailure(503)
        pool = AsyncAPIPool(self.ledger, sleep=sleep)
        with self.assertRaises(HTTPFailure):
            await pool.request("failure", .01, send, lambda r: (0, {}), events.append)
        self.assertEqual(len(events), 7)
        self.assertEqual(len(delays), 6)
        self.assertEqual(self.ledger.snapshot()["reserved"], .07)
        for attempt, delay in enumerate(delays):
            self.assertGreaterEqual(delay, 2 ** attempt / 2)
            self.assertLessEqual(delay, 2 ** attempt)

    async def test_429_then_success_keeps_ambiguous_first_charge(self):
        attempts = 0
        async def send():
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise HTTPFailure(429)
            return {"tokens": 2}
        async def sleep(_):
            pass
        pool = AsyncAPIPool(self.ledger, sleep=sleep)
        result = await pool.request("recover", .01, send, lambda r: (.002, r))
        self.assertEqual(result, {"tokens": 2})
        self.assertEqual(self.ledger.snapshot()["spent"], .002)
        self.assertEqual(self.ledger.snapshot()["reserved"], .01)

    async def test_budget_stops_before_send(self):
        self.ledger.reserve("other", 25)
        called = False
        async def send():
            nonlocal called
            called = True
        with self.assertRaises(BudgetExceeded):
            await AsyncAPIPool(self.ledger).request("blocked", 1, send, lambda _: (0, {}))
        self.assertFalse(called)

    async def test_non_retryable_failure_is_retained_and_reported(self):
        events = []
        async def send():
            raise HTTPFailure(400)
        with self.assertRaises(HTTPFailure):
            await AsyncAPIPool(self.ledger).request("bad", .01, send, lambda r: (0, {}), events.append)
        self.assertEqual(len(events), 1)
        self.assertEqual(self.ledger.snapshot()["reserved"], .01)

    async def test_status_writes_immediately_and_on_stop(self):
        stop = asyncio.Event()
        rows = [{"benchmark": "official", "arm": "bare", "done": 1, "total": 3, "errors": 1}]
        task = asyncio.create_task(status_loop(self.root / "STATUS.md", lambda: rows,
                                               self.ledger, stop, interval=.005,
                                               started_at=time.monotonic() - 60))
        await asyncio.sleep(.01)
        rows[0]["done"] = 3
        stop.set()
        await task
        status = (self.root / "STATUS.md").read_text()
        self.assertIn("| official | bare | 3 | 3 | 1 | complete |", status)
        self.assertIn("API spend: $0.000000", status)


if __name__ == "__main__":
    unittest.main()
