"""Shared controls for one asyncio campaign, with a process-safe spend ledger.

The SQLite ledger is shared across processes. API concurrency is shared across all
AsyncAPIPool instances on a single event loop; launch the campaign in one process
and one loop to enforce the global 40-request limit. A reservation is a caller's
verified *upper bound* on an attempt's entire charge, including output tokens.
Unknown failures remain encumbered until usage is reconciled. This can stop a run
early, but cannot turn an unknown charge into available budget.
"""
from __future__ import annotations

import asyncio
import contextlib
from datetime import datetime, timezone
from decimal import Decimal, ROUND_CEILING
import fcntl
import hashlib
import inspect
import json
import os
from pathlib import Path
import random
import sqlite3
import time
from typing import Any, Awaitable, Callable, Iterable, Mapping, Sequence
import weakref

NANODOLLARS = Decimal(1_000_000_000)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _money(value: Any) -> int:
    number = Decimal(str(value))
    if not number.is_finite() or number < 0:
        raise ValueError("cost must be finite and nonnegative")
    return int((number * NANODOLLARS).to_integral_value(rounding=ROUND_CEILING))


class BudgetExceeded(RuntimeError):
    """A new bounded attempt would exhaust the campaign's remaining budget."""


class ReservationExists(RuntimeError):
    """An attempt identity was already used; reconcile rather than resend it."""


class BudgetLedger:
    def __init__(self, path: str | Path, limit: Any = 25) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.limit_units = _money(limit)
        if not 0 < self.limit_units <= _money(25):
            raise ValueError("campaign limit must be positive and at most $25")
        with self._connect() as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.execute("CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
            db.execute("""CREATE TABLE IF NOT EXISTS calls (
                id TEXT PRIMARY KEY, reserved INTEGER NOT NULL, actual INTEGER,
                state TEXT NOT NULL, usage TEXT, created TEXT NOT NULL, updated TEXT NOT NULL)""")
            db.execute("""CREATE TABLE IF NOT EXISTS budget_epochs (
                id TEXT PRIMARY KEY, limit_units INTEGER NOT NULL, starts_at TEXT NOT NULL,
                benchmark TEXT NOT NULL, authority_id TEXT NOT NULL, created TEXT NOT NULL)""")
            db.execute("""CREATE TABLE IF NOT EXISTS epoch_calls (
                call_id TEXT PRIMARY KEY, epoch_id TEXT NOT NULL,
                FOREIGN KEY(call_id) REFERENCES calls(id),
                FOREIGN KEY(epoch_id) REFERENCES budget_epochs(id))""")
            db.execute("BEGIN IMMEDIATE")
            db.execute("INSERT OR IGNORE INTO settings VALUES ('limit', ?)", (str(self.limit_units),))
            found = int(db.execute("SELECT value FROM settings WHERE key='limit'").fetchone()[0])
            if found != self.limit_units:
                raise ValueError("existing ledger has a different immutable spending limit")

    @contextlib.contextmanager
    def _connect(self):
        db = sqlite3.connect(self.path, timeout=30)
        try:
            db.execute("PRAGMA busy_timeout=30000")
            db.execute("PRAGMA synchronous=FULL")
            with db:
                yield db
        finally:
            db.close()

    @staticmethod
    def _totals(db: sqlite3.Connection, epoch_id=None) -> tuple[int, int]:
        query="""SELECT
            COALESCE(SUM(CASE WHEN state='settled' THEN actual ELSE 0 END),0),
            COALESCE(SUM(CASE WHEN state='reserved' THEN reserved ELSE 0 END),0)
            FROM calls"""
        params=()
        if epoch_id is not None:
            query+=" JOIN epoch_calls ON epoch_calls.call_id=calls.id WHERE epoch_calls.epoch_id=?"
            params=(epoch_id,)
        spent, reserved = db.execute(query,params).fetchone()
        return int(spent), int(reserved)

    def _context(self, db):
        active=db.execute("SELECT value FROM settings WHERE key='active_epoch'").fetchone()
        if not active:
            return None,self.limit_units,None,None,None
        epoch=db.execute("SELECT id,limit_units,benchmark,starts_at,authority_id FROM budget_epochs WHERE id=?",(active[0],)).fetchone()
        if epoch is None:
            raise BudgetExceeded("active budget epoch is missing; fail closed")
        return tuple(epoch)

    def authorize_epoch(self, *, epoch_id, limit, starts_at, benchmark, authority_id):
        """Record a fresh explicit owner allowance without changing old calls.

        Activation is idempotent for the identical authorization. A different
        active authorization requires a separate owner instruction, never a
        restart. Any calls already made since the owner's timestamp are included.
        """
        units=_money(limit)
        if not epoch_id or not authority_id or benchmark!='longmemeval' or not 0<units<=_money(21):
            raise ValueError("explicit LongMemEval owner authorization of at most $21 is required")
        anchor=datetime.fromisoformat(starts_at.replace('Z','+00:00'))
        if anchor.tzinfo is None or anchor>datetime.now(timezone.utc):
            raise ValueError("budget epoch requires a past/current timezone-aware owner timestamp")
        starts_at=anchor.astimezone(timezone.utc).isoformat()
        expected=(epoch_id,units,starts_at,benchmark,authority_id)
        with self._connect() as db:
            db.execute('BEGIN IMMEDIATE')
            active=db.execute("SELECT value FROM settings WHERE key='active_epoch'").fetchone()
            if active:
                old=db.execute('SELECT id,limit_units,starts_at,benchmark,authority_id FROM budget_epochs WHERE id=?',(active[0],)).fetchone()
                if old!=expected:
                    raise ValueError("another budget epoch is already active; never reset it on restart")
                return self.snapshot()
            candidates=db.execute('SELECT id,created FROM calls').fetchall()
            scoped=[identity for identity,created in candidates
                    if datetime.fromisoformat(created.replace('Z','+00:00'))>=anchor]
            if any(not identity.startswith(benchmark+'/') for identity in scoped):
                raise ValueError("calls outside LongMemEval exist after the owner budget timestamp")
            db.execute('INSERT INTO budget_epochs VALUES (?,?,?,?,?,?)',(*expected,utc_now()))
            db.executemany('INSERT INTO epoch_calls VALUES (?,?)',[(identity,epoch_id) for identity in scoped])
            spent,reserved=self._totals(db,epoch_id)
            if spent+reserved>units:
                raise BudgetExceeded("newly authorized budget is already exhausted")
            db.execute("INSERT INTO settings VALUES ('active_epoch',?)",(epoch_id,))
        return self.snapshot()

    def reserve(self, id: str, upper_cost: Any) -> bool:
        """Atomically reserve a unique attempt; raise before sending if unavailable."""
        units = _money(upper_cost)
        if not id or units <= 0:
            raise ValueError("attempt id and a positive verified upper cost are required")
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            if db.execute("SELECT 1 FROM settings WHERE key='bound_violation'").fetchone():
                raise BudgetExceeded("campaign halted after an attempt bound violation")
            if db.execute("SELECT 1 FROM calls WHERE id=?", (id,)).fetchone():
                raise ReservationExists(f"attempt already recorded: {id}")
            epoch_id,limit_units,benchmark,_,_=self._context(db)
            if benchmark and not id.startswith(benchmark+'/'):
                raise BudgetExceeded("owner budget is restricted to LongMemEval-S")
            spent, outstanding = self._totals(db,epoch_id)
            if spent + outstanding + units > limit_units:
                raise BudgetExceeded("authorized campaign budget has insufficient unencumbered funds")
            now = utc_now()
            db.execute("INSERT INTO calls VALUES (?, ?, NULL, 'reserved', NULL, ?, ?)",
                       (id, units, now, now))
            if epoch_id:
                db.execute('INSERT INTO epoch_calls VALUES (?,?)',(id,epoch_id))
        return True

    def settle(self, id: str, actual_cost: Any, usage: Mapping[str, Any]) -> None:
        """Reconcile usage; a bound violation is recorded and stops further calls."""
        units = _money(actual_cost)
        encoded = canonical(dict(usage))
        overrun = False
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT reserved, actual, state, usage FROM calls WHERE id=?", (id,)).fetchone()
            if row is None:
                raise KeyError(f"unknown reservation {id}")
            reserved, actual, state, old_usage = row
            if state == "settled":
                if actual == units and old_usage == encoded:
                    return
                raise ValueError("cannot replace previously settled usage")
            if state != "reserved":
                raise ValueError("cannot charge an attempt already confirmed unbilled")
            overrun = units > reserved
            db.execute("UPDATE calls SET actual=?, state='settled', usage=?, updated=? WHERE id=?",
                       (units, encoded, utc_now(), id))
            if overrun:
                db.execute("INSERT OR REPLACE INTO settings VALUES ('bound_violation', ?)", (id,))
        if overrun:
            raise BudgetExceeded("verified attempt upper bound was violated; actual charge recorded, halt campaign")

    def confirm_unbilled(self, id: str, evidence: Mapping[str, Any]) -> None:
        """Release only after affirmative provider evidence of a zero charge."""
        if not evidence:
            raise ValueError("provider evidence is required")
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT state FROM calls WHERE id=?", (id,)).fetchone()
            if row is None:
                raise KeyError(id)
            if row[0] != "reserved":
                raise ValueError("only an outstanding reservation can be confirmed unbilled")
            db.execute("UPDATE calls SET state='unbilled', actual=0, usage=?, updated=? WHERE id=?",
                       (canonical(dict(evidence)), utc_now(), id))

    def snapshot(self) -> dict[str, Any]:
        with self._connect() as db:
            epoch_id,limit_units,benchmark,starts_at,authority_id=self._context(db)
            spent, reserved = self._totals(db,epoch_id)
            lifetime_spent,lifetime_reserved=self._totals(db)
            counts = dict(db.execute("SELECT state, COUNT(*) FROM calls GROUP BY state").fetchall())
            violation = db.execute("SELECT value FROM settings WHERE key='bound_violation'").fetchone()
        return {"limit": limit_units / 1e9, "spent": spent / 1e9,
                "reserved": reserved / 1e9, "committed": (spent + reserved) / 1e9,
                "remaining": max(0, limit_units - spent - reserved) / 1e9,
                "calls": counts, "bound_violation": violation[0] if violation else None,
                "epoch_id":epoch_id,"benchmark":benchmark,"starts_at":starts_at,"authority_id":authority_id,
                "lifetime_spent":lifetime_spent/1e9,"lifetime_reserved":lifetime_reserved/1e9,
                "historical_spent":(lifetime_spent-spent)/1e9,
                "historical_reserved":(lifetime_reserved-reserved)/1e9}


_loop_semaphores: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()


def _global_semaphore() -> asyncio.Semaphore:
    loop = asyncio.get_running_loop()
    if loop not in _loop_semaphores:
        _loop_semaphores[loop] = asyncio.Semaphore(40)
    return _loop_semaphores[loop]


def http_status(exc: BaseException) -> int | None:
    for candidate in (getattr(exc, "status_code", None), getattr(exc, "status", None),
                      getattr(getattr(exc, "response", None), "status_code", None)):
        if isinstance(candidate, int):
            return candidate
    return None


class AsyncAPIPool:
    """Bounded provider attempts; retries also consume bounded reservations.

    ``send`` takes no arguments and returns a response. ``account(response)``
    returns ``(actual_dollars, usage_dict)`` from provider usage fields. A failed
    attempt stays reserved, including a 429/5xx, unless the caller later provides
    affirmative zero-charge evidence. No requests are issued by this module.
    """
    def __init__(self, ledger: BudgetLedger, max_inflight: int = 40,
                 max_retries: int = 6, base_delay: float = 1,
                 max_delay: float = 60, seed: int = 0,
                 sleep: Callable[[float], Awaitable[None]] = asyncio.sleep) -> None:
        if not 1 <= max_inflight <= 40 or not 0 <= max_retries <= 6:
            raise ValueError("max_inflight must be 1..40 and max_retries 0..6")
        self.ledger = ledger
        self.semaphore = asyncio.Semaphore(max_inflight)
        self.max_retries = max_retries
        self.base_delay, self.max_delay = base_delay, max_delay
        self.random = random.Random(seed)
        self.sleep = sleep

    async def request(self, call_id: str, upper_cost: Any,
                      send: Callable[[], Awaitable[Any]],
                      account: Callable[[Any], tuple[Any, Mapping[str, Any]]],
                      on_attempt: Callable[[Mapping[str, Any]], Any] | None = None) -> Any:
        for attempt in range(self.max_retries + 1):
            attempt_id = f"{call_id}/attempt-{attempt}"
            try:
                async with self.semaphore, _global_semaphore():
                    if self.ledger.snapshot()["bound_violation"]:
                        raise BudgetExceeded("campaign stopped after an upper-bound violation")
                    self.ledger.reserve(attempt_id, upper_cost)
                    response = await send()
                    actual_cost, usage = account(response)
                    self.ledger.settle(attempt_id, actual_cost, usage)
                if on_attempt:
                    value = on_attempt({"id": attempt_id, "attempt": attempt, "status": "complete",
                                        "cost": actual_cost, "usage": dict(usage), "timestamp": utc_now()})
                    if inspect.isawaitable(value):
                        await value
                return response
            except (BudgetExceeded, ReservationExists):
                raise
            except Exception as exc:
                status = http_status(exc)
                if on_attempt:
                    value = on_attempt({"id": attempt_id, "attempt": attempt, "status": "error",
                                        "http_status": status, "error_type": type(exc).__name__,
                                        "timestamp": utc_now(), "charge_state": "reserved_until_reconciled"})
                    if inspect.isawaitable(value):
                        await value
                import httpx
                transient = isinstance(exc, (httpx.TimeoutException, httpx.NetworkError, httpx.RemoteProtocolError))
                if not transient and status != 429 and not (status is not None and 500 <= status <= 599):
                    raise
                if attempt == self.max_retries:
                    raise
                ceiling = min(self.max_delay, self.base_delay * (2 ** attempt))
                await self.sleep(self.random.uniform(ceiling / 2, ceiling))
        raise AssertionError("unreachable retry state")


def shard_id(item_id: str, workers: int = 10) -> int:
    if workers <= 0:
        raise ValueError("workers must be positive")
    return int.from_bytes(hashlib.sha256(str(item_id).encode("utf-8")).digest(), "big") % workers


def write_shard_map(path: str | Path, item_ids: Iterable[str], workers: int = 10,
                    seed: int = 0) -> dict[str, Any]:
    ids = list(item_ids)
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate official item ids")
    manifest = {"algorithm": "int.from_bytes(sha256(utf8(item_id)), 'big') % workers",
                "seed": seed, "seed_role": "recorded trial/sample seed; hash routing is seed-independent",
                "workers": workers, "items": {i: shard_id(i, workers) for i in sorted(ids)}}
    atomic_write(path, canonical(manifest) + "\n")
    return manifest


def atomic_write(path: str | Path, content: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    with temporary.open("x", encoding="utf-8") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    directory = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def append_jsonl(path: str | Path, row: Mapping[str, Any]) -> None:
    """Serialize concurrent appends under a file lock, and flush before returning."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            handle.write(canonical(dict(row)) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        return []
    rows = []
    with path.open(encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_SH)
        for line_number, line in enumerate(handle, 1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"corrupt checkpoint {path}:{line_number}; preserve and reconcile it") from exc
            if not isinstance(row, dict):
                raise ValueError(f"checkpoint row must be an object: {path}:{line_number}")
            rows.append(row)
    return rows


def record_key(row: Mapping[str, Any], key_fields: Sequence[str] = ("item_id", "trial")) -> str:
    return canonical([row[field] for field in key_fields])


class Checkpoint:
    """Successful item records only; failed attempts have a separate append log.

    Identity must include dataset, prompt, model and engine hashes so restarting
    with altered inputs cannot reuse an older success. Caller supplies item/trial
    keys. A torn final line is reported as a failure instead of silently ignored.
    """
    def __init__(self, path: str | Path, identity: Mapping[str, Any],
                 key_fields: Sequence[str] = ("item_id", "trial")) -> None:
        self.path = Path(path)
        self.identity = hashlib.sha256(canonical(dict(identity)).encode()).hexdigest()
        self.key_fields = tuple(key_fields)
        self.done: dict[str, dict[str, Any]] = {}
        for row in read_jsonl(self.path):
            if row.get("run_identity") != self.identity:
                raise ValueError("checkpoint identity mismatch: dataset/prompt/model/engine changed")
            key = record_key(row, self.key_fields)
            if key in self.done:
                raise ValueError("duplicate completed item in checkpoint")
            self.done[key] = row

    def contains(self, row: Mapping[str, Any]) -> bool:
        return record_key(row, self.key_fields) in self.done

    def append(self, row: Mapping[str, Any]) -> None:
        key = record_key(row, self.key_fields)
        if key in self.done:
            raise ValueError("duplicate completed item")
        if row.get("status", "complete") != "complete":
            raise ValueError("failed attempts belong in append_failure, not completed checkpoints")
        result = dict(row)
        result["run_identity"] = self.identity
        result.setdefault("timestamp", utc_now())
        append_jsonl(self.path, result)
        self.done[key] = result

    def append_failure(self, row: Mapping[str, Any]) -> None:
        result = dict(row)
        result.update(run_identity=self.identity, status="error")
        result.setdefault("timestamp", utc_now())
        append_jsonl(self.path.with_suffix(".errors.jsonl"), result)


def strict_merge(paths: Iterable[str | Path], expected_keys: Iterable[str],
                 output: str | Path, key_fields: Sequence[str] = ("item_id", "trial"),
                 identity: str | None = None) -> list[dict[str, Any]]:
    """Write merged predictions only after exact coverage and uniqueness checks.

    expected_keys are record_key(...) strings. Official scoring is invoked by the
    caller once this returns; no benchmark scorer is implemented or modified here.
    """
    keys = list(expected_keys)
    if len(set(keys)) != len(keys):
        raise ValueError("expected official set contains duplicate item/trial identities")
    expected = set(keys)
    merged: dict[str, dict[str, Any]] = {}
    for path in paths:
        for row in read_jsonl(path):
            if row.get("status", "complete") != "complete":
                raise ValueError("failed item cannot enter official scoring")
            if identity is not None and row.get("run_identity") != identity:
                raise ValueError("merge run identity mismatch")
            key = record_key(row, key_fields)
            if key in merged:
                raise ValueError(f"duplicate item/trial across shards: {key}")
            merged[key] = row
    missing, extra = expected - merged.keys(), merged.keys() - expected
    if missing or extra:
        raise ValueError(f"coverage mismatch: missing={len(missing)}, unexpected={len(extra)}")
    rows = [merged[key] for key in keys]
    atomic_write(output, "".join(canonical(row) + "\n" for row in rows))
    return rows


def write_status(path: str | Path, progress: Iterable[Mapping[str, Any]],
                 ledger: BudgetLedger, started_at: float) -> None:
    """Write a current status snapshot; started_at is time.monotonic()."""
    rows = list(progress)
    snapshot = ledger.snapshot()
    elapsed = max(0, time.monotonic() - started_at)
    lines = ["# Benchmark campaign status", "", f"Updated: {utc_now()}", "",
             f"API spend: ${snapshot['spent']:.6f} / ${snapshot['limit']:.2f}; "
             f"outstanding reservations: ${snapshot['reserved']:.6f}; "
             f"available: ${snapshot['remaining']:.6f}.", "",
             "| Benchmark | Arm | Done | Full official total | Item errors | ETA | Stage | API attempt errors |", "|---|---|---:|---:|---:|---|---|---:|"]
    for row in rows:
        done, total = int(row["done"]), int(row["total"])
        if not 0 <= done <= total:
            raise ValueError("status done/total is invalid")
        eta = "complete" if done == total else (
            f"{elapsed * (total - done) / done / 60:.1f} min" if done and elapsed else "pending")
        benchmark, arm = str(row["benchmark"]).replace("|", "\\|"), str(row["arm"]).replace("|", "\\|")
        lines.append(f"| {benchmark} | {arm} | {done} | {total} | {int(row.get('errors', 0))} | {eta} | {row.get('stage','unspecified')} | {int(row.get('api_errors',0))} |")
    lines += ["", "ETA uses observed completions since campaign start; setup and changing task lengths can shift it.",
              "Outstanding reservations include ambiguous failed attempts until provider usage is reconciled.", ""]
    atomic_write(path, "\n".join(lines))


async def status_loop(path: str | Path, progress: Callable[[], Iterable[Mapping[str, Any]]],
                      ledger: BudgetLedger, stop: asyncio.Event, interval: float = 300,
                      started_at: float | None = None) -> None:
    started_at = time.monotonic() if started_at is None else started_at
    while True:
        write_status(path, progress(), ledger, started_at)
        if stop.is_set():
            return
        try:
            await asyncio.wait_for(stop.wait(), timeout=interval)
        except asyncio.TimeoutError:
            pass
