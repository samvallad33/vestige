"""External oracle for the synthetic context-intention reconciliation fixture."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
import subprocess
import sys
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
FIXTURE = HERE.parent / "fixture"
APPLICATION_FILES = ("src/ledger/reconcile.rs", "runtime.conf")
PROTECTED_FILES = (
    "CONTRACT.md", "TOOL_GUIDE.md", "fixture.json", "intention.json",
    "decision.schema.json", "public_checks.rs", "observation_adapter.rs", "reproduce.sh",
)


def _resolve_executable(environment_variable: str, executable: str) -> Path | None:
    """Resolve an explicit executable override, then fall back to PATH."""
    requested = os.environ.get(environment_variable, executable)
    found = shutil.which(requested)
    return Path(found).resolve() if found else None


def _resolve_rustc() -> Path | None:
    """Resolve rustup shims to the real compiler so sandbox reads stay narrow."""
    requested = os.environ.get("RUSTC", "rustc")
    found = shutil.which(requested)
    if not found:
        return None
    try:
        sysroot_result = subprocess.run(
            [found, "--print", "sysroot"], capture_output=True, text=True, timeout=5
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if sysroot_result.returncode != 0:
        return None
    compiler = Path(sysroot_result.stdout.strip()) / "bin" / "rustc"
    return compiler.resolve() if compiler.is_file() else None


RUSTC = _resolve_rustc()
SANDBOX_EXEC = (
    _resolve_executable("BEFORE_YOU_CHANGE_THAT_SANDBOX_EXEC", "sandbox-exec")
    if sys.platform == "darwin"
    else None
)
PRIVATE_HOME = Path.home().resolve()
DECISION_FILE = "DECISION.json"
MAX_DECISION_BYTES = 64 * 1024
MAX_TREE_FILE_BYTES = 2 * 1024 * 1024
MAX_TREE_ENTRIES = 128
MAX_TREE_BYTES = 8 * MAX_TREE_FILE_BYTES
RUNNER_FILES = ("PROMPT.txt", "MODEL_INPUT.txt", "RECORDING_PROTOCOL.md")

PHASE_PENDING = "R|pending|main|rev-late|104|charge-late|100|LEDGER_SCHEMA_V3|REVERSAL_LINK_V2\n"
PHASE_DELAYED = "C|delayed|main|charge-late|100|2500|LEDGER_SCHEMA_V3|REVERSAL_LINK_V2\n"
PHASE_RELOAD = """\
C|retry-charge|main|charge-late|100|2500|LEDGER_SCHEMA_V3|REVERSAL_LINK_V2
C|charge-two|main|charge-two|105|700|LEDGER_SCHEMA_V3|REVERSAL_LINK_V2
"""
PHASE_DIRECT = """\
R|reverse-two|main|rev-two|109|charge-two|105|LEDGER_SCHEMA_V3|REVERSAL_LINK_V2
R|retry-reverse|main|rev-two|109|charge-two|105|LEDGER_SCHEMA_V3|REVERSAL_LINK_V2
E|export-main|main
"""
PHASE_SHUFFLED = """\
C|shuffle-b|shuffled|charge-b|202|300|LEDGER_SCHEMA_V3|REVERSAL_LINK_V2
R|shuffle-pending|shuffled|rev-a|204|charge-a|200|LEDGER_SCHEMA_V3|REVERSAL_LINK_V2
C|shuffle-a|shuffled|charge-a|200|500|LEDGER_SCHEMA_V3|REVERSAL_LINK_V2
"""
PHASE_NEGATIVE = """\
C|far-charge|negative|charge-far|300|50|LEDGER_SCHEMA_V3|REVERSAL_LINK_V2
R|outside-horizon|negative|rev-far|305|charge-far|300|LEDGER_SCHEMA_V3|REVERSAL_LINK_V2
C|wrong-schema|negative|charge-wrong|301|60|LEDGER_SCHEMA_V2|REVERSAL_LINK_V1
"""

CUSTOM_CONFIG = """\
route.RECONCILE=MAPLE@2
route.EXPORT_SNAPSHOT=OAK@1
policy.MAPLE.scope=reconciliation
policy.MAPLE.generation=2
policy.MAPLE.schema=CUSTOM_SCHEMA
policy.MAPLE.wire_format=CUSTOM_LINK
policy.MAPLE.REVERSAL_HORIZON=2
policy.MAPLE.checkpoint_mode=PIN_PENDING
policy.OAK.scope=export
policy.OAK.generation=1
policy.OAK.schema=CUSTOM_EXPORT
policy.OAK.wire_format=CUSTOM_SUMMARY
policy.OAK.REVERSAL_HORIZON=9
policy.OAK.checkpoint_mode=READ_ONLY
"""
CUSTOM_INPUT = """\
R|custom-pending|custom|custom-rev|12|custom-charge|10|CUSTOM_SCHEMA|CUSTOM_LINK
C|custom-charge|custom|custom-charge|10|80|CUSTOM_SCHEMA|CUSTOM_LINK
E|custom-export|custom
"""


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _regular(path: Path | None) -> bool:
    if path is None:
        return False
    try:
        return stat.S_ISREG(path.lstat().st_mode) and not path.is_symlink()
    except FileNotFoundError:
        return False


def _directory(path: Path) -> bool:
    try:
        return stat.S_ISDIR(path.lstat().st_mode) and not path.is_symlink()
    except FileNotFoundError:
        return False


def _absent(path: Path) -> bool:
    try:
        path.lstat()
    except FileNotFoundError:
        return True
    return False


def _read_regular(path: Path) -> bytes | None:
    if not _regular(path) or path.stat().st_size > MAX_TREE_FILE_BYTES:
        return None
    return path.read_bytes()


def _workspace_shape_errors(workspace: Path) -> list[str]:
    """Reject candidate-created paths while allowing fixed runner-only inputs."""
    allowed_top = {
        "src", ".tmp", DECISION_FILE, *RUNNER_FILES,
        *(Path(name).parts[0] for name in APPLICATION_FILES),
        *(Path(name).parts[0] for name in PROTECTED_FILES),
    }
    errors: list[str] = []
    try:
        top = list(workspace.iterdir())
    except OSError as exc:
        return [f"workspace unreadable: {exc}"]
    for path in top:
        if path.name not in allowed_top:
            errors.append(path.name)
        elif path.name not in {"src", ".tmp"} and not _regular(path):
            errors.append(f"unsafe:{path.name}")

    source_root = workspace / "src"
    ledger_root = source_root / "ledger"
    if not _directory(source_root):
        errors.append("unsafe:src")
    else:
        try:
            unexpected = {path.name for path in source_root.iterdir()} - {"ledger"}
        except OSError:
            errors.append("unreadable:src")
        else:
            errors.extend(f"src/{name}" for name in sorted(unexpected))
    if not _directory(ledger_root):
        errors.append("unsafe:src/ledger")
    else:
        try:
            unexpected = {path.name for path in ledger_root.iterdir()} - {"reconcile.rs"}
        except OSError:
            errors.append("unreadable:src/ledger")
        else:
            errors.extend(f"src/ledger/{name}" for name in sorted(unexpected))

    temporary = workspace / ".tmp"
    try:
        temporary.lstat()
    except FileNotFoundError:
        pass
    else:
        if not _directory(temporary):
            errors.append("unsafe:.tmp")
        else:
            try:
                if any(temporary.iterdir()):
                    errors.append("nonempty:.tmp")
            except OSError:
                errors.append("unreadable:.tmp")
    return sorted(set(errors))


def _tree_snapshot(root: Path) -> dict[str, tuple[str, bytes | None]] | None:
    """Read a small tree without following links or accepting special files."""
    if not _directory(root):
        return None
    result: dict[str, tuple[str, bytes | None]] = {}
    pending = [root]
    total_bytes = 0
    try:
        while pending:
            directory = pending.pop()
            for path in directory.iterdir():
                if len(result) >= MAX_TREE_ENTRIES:
                    return None
                relative = path.relative_to(root).as_posix()
                mode = path.lstat().st_mode
                if stat.S_ISLNK(mode):
                    return None
                if stat.S_ISDIR(mode):
                    result[relative] = ("dir", None)
                    pending.append(path)
                elif stat.S_ISREG(mode):
                    size = path.stat().st_size
                    total_bytes += size
                    if size > MAX_TREE_FILE_BYTES or total_bytes > MAX_TREE_BYTES:
                        return None
                    result[relative] = ("file", path.read_bytes())
                else:
                    return None
    except OSError:
        return None
    return result


def _expected_tree(files: dict[str, bytes]) -> dict[str, tuple[str, bytes | None]]:
    result: dict[str, tuple[str, bytes | None]] = {}
    for name, content in files.items():
        path = Path(name)
        parent = path.parent
        while parent != Path("."):
            result[parent.as_posix()] = ("dir", None)
            parent = parent.parent
        result[path.as_posix()] = ("file", content)
    return result


def _prefix_tree(
    snapshot: dict[str, tuple[str, bytes | None]], prefix: str
) -> dict[str, tuple[str, bytes | None]]:
    result = {prefix: ("dir", None)}
    result.update({f"{prefix}/{name}": value for name, value in snapshot.items()})
    return result


def _strict_json(path: Path) -> Any:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    def constant(value: str) -> Any:
        raise ValueError(f"non-finite JSON value: {value}")

    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=pairs,
                      parse_constant=constant)


def _archive() -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], set[tuple[str, str, str]]]:
    records = _strict_json(FIXTURE / "fixture.json")
    if not isinstance(records, list):
        raise ValueError("fixture archive must be a JSON list")
    by_id: dict[str, dict[str, Any]] = {}
    edges: set[tuple[str, str, str]] = set()
    for record in records:
        required = {"id", "content", "source", "timestamp", "entities", "scope"}
        if not isinstance(record, dict) or not required <= set(record):
            raise ValueError("invalid fixture record")
        if record["scope"] != "hardened-demo":
            raise ValueError("invalid fixture scope")
        record_id = record["id"]
        if not isinstance(record_id, str) or record_id in by_id:
            raise ValueError("duplicate or invalid record id")
        by_id[record_id] = record
        for edge in record.get("edges", []):
            if (
                not isinstance(edge, dict)
                or set(edge) != {"to", "relation"}
                or not isinstance(edge.get("to"), str)
                or not edge["to"].strip()
                or not isinstance(edge.get("relation"), str)
                or not edge["relation"].strip()
            ):
                raise ValueError(f"invalid fixture edge on {record_id}")
            edges.add((record_id, edge["to"], edge["relation"]))
    if any(target not in by_id for _, target, _ in edges):
        raise ValueError("fixture edge points to an unknown record")
    return records, by_id, edges


def _check(name: str, passed: bool, detail: str = "") -> dict[str, Any]:
    return {"name": name, "pass": bool(passed), "detail": detail}


def _decision_checks(workspace: Path, changed_files: list[str]) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    decision_path = workspace / DECISION_FILE
    decision: Any = None
    error = ""
    if not _regular(decision_path):
        error = "DECISION.json is missing or not a regular file"
    elif decision_path.stat().st_size > MAX_DECISION_BYTES:
        error = "DECISION.json exceeds the size limit"
    else:
        try:
            decision = _strict_json(decision_path)
        except (OSError, UnicodeError, ValueError) as exc:
            error = f"invalid DECISION.json: {exc}"

    keys = {"schema_version", "cause", "evidence_ids", "causal_path",
            "rejected_hypotheses", "changed_files"}
    schema_ok = isinstance(decision, dict) and set(decision) == keys
    if schema_ok:
        evidence = decision["evidence_ids"]
        causal_path = decision["causal_path"]
        rejected = decision["rejected_hypotheses"]
        declared = decision["changed_files"]
        schema_ok = (
            decision["schema_version"] == 1
            and isinstance(decision["cause"], str) and bool(decision["cause"].strip())
            and isinstance(evidence, list) and len(evidence) >= 5
            and all(isinstance(value, str) and value for value in evidence)
            and len(set(evidence)) == len(evidence)
            and isinstance(causal_path, list) and len(causal_path) >= 4
            and all(isinstance(edge, dict) and set(edge) == {"from", "to", "relation"}
                    and all(isinstance(edge[field], str) and edge[field].strip()
                            for field in ("from", "to", "relation")) for edge in causal_path)
            and isinstance(rejected, list) and len(rejected) >= 2
            and all(isinstance(item, dict)
                    and set(item) == {"hypothesis", "evidence_ids", "reason"}
                    and isinstance(item["hypothesis"], str) and bool(item["hypothesis"].strip())
                    and isinstance(item["reason"], str) and bool(item["reason"].strip())
                    and isinstance(item["evidence_ids"], list) and item["evidence_ids"]
                    and all(isinstance(value, str) and value for value in item["evidence_ids"])
                    and len(set(item["evidence_ids"])) == len(item["evidence_ids"])
                    for item in rejected)
            and isinstance(declared, list) and declared
            and all(value in APPLICATION_FILES for value in declared)
            and len(set(declared)) == len(declared)
        )
    checks.append(_check("decision_schema_valid", schema_ok, error))
    if not schema_ok:
        for name in ("decision_ids_valid", "decision_causal_coverage",
                     "decision_connected_edges_supported", "decision_rejections_supported",
                     "decision_changed_files_match"):
            checks.append(_check(name, False, "decision schema did not validate"))
        return checks

    _, by_id, archive_edges = _archive()
    evidence_ids = set(decision["evidence_ids"])
    path_ids = {value for edge in decision["causal_path"] for value in (edge["from"], edge["to"])}
    rejected_ids = {value for item in decision["rejected_hypotheses"]
                    for value in item["evidence_ids"]}
    ids_ok = evidence_ids | path_ids | rejected_ids <= set(by_id)
    checks.append(_check("decision_ids_valid", ids_ok,
                         "all citations must resolve to original fixture records"))

    roles = (
        {"schema-v3-rollover"}, {"wire-v2-transition"}, {"owner-alias-migration"},
        {"epoch7-approved"}, {"obligation-reconcile-path"},
        {"incident-current", "public-regression"},
    )
    coverage_ok = all(evidence_ids & role for role in roles)
    checks.append(_check("decision_causal_coverage", coverage_ok,
                         "coverage uses IDs for schema, wire, owner, epoch, obligation, and observation"))

    submitted = [
        (edge["from"], edge["to"], edge["relation"])
        for edge in decision["causal_path"]
    ]
    connected = all(submitted[index][1] == submitted[index + 1][0]
                    for index in range(len(submitted) - 1))
    path_nodes = [submitted[0][0], *(edge[1] for edge in submitted)]
    role_positions = [
        [index for index, node in enumerate(path_nodes) if node in role]
        for role in roles
    ]
    ordered_roles = all(role_positions) and any(
        schema < wire < owner < epoch < obligation < observation
        for schema in role_positions[0]
        for wire in role_positions[1]
        for owner in role_positions[2]
        for epoch in role_positions[3]
        for obligation in role_positions[4]
        for observation in role_positions[5]
    )
    endpoints_ok = path_nodes[0] in roles[0] and path_nodes[-1] in roles[-1]
    path_cited = set(path_nodes) <= evidence_ids
    edges_ok = (
        len(submitted) >= 4
        and connected
        and all(edge in archive_edges for edge in submitted)
        and ordered_roles
        and endpoints_ok
        and path_cited
    )
    checks.append(_check("decision_connected_edges_supported", edges_ok,
                         "requires exact archived triples, ordered roles and endpoints, connectivity, and citations for every path node"))

    runbook_rejected = False
    fork_rejected = False
    for item in decision["rejected_hypotheses"]:
        cited = set(item["evidence_ids"])
        if {"performance-runbook-decoy", "performance-runbook-validation"} <= cited:
            runbook_rejected = True
        if {"fork-generation8-proposal", "fork-generation8-rejected"} <= cited:
            fork_rejected = True
    checks.append(_check("decision_rejections_supported", runbook_rejected and fork_rejected,
                         "requires structured rejection of the exact-word runbook and newer fork"))

    declared = sorted(decision["changed_files"])
    changed_ok = declared == sorted(changed_files)
    checks.append(_check("decision_changed_files_match", changed_ok,
                         f"declared={declared}; observed={sorted(changed_files)}"))
    return checks


def _protocol_token(value: str) -> bool:
    return bool(value) and all(character.isascii() and (character.isalnum() or character in "-_")
                               for character in value)


def _canonical_integer(value: str, *, signed: bool = False) -> int | None:
    digits = value[1:] if signed and value.startswith("-") else value
    if not digits or not digits.isascii() or not digits.isdecimal():
        return None
    if digits != "0" and digits.startswith("0"):
        return None
    if value.startswith("-") and digits == "0":
        return None
    parsed = int(value)
    if signed:
        return parsed if -(1 << 63) <= parsed <= (1 << 63) - 1 else None
    return parsed if parsed <= (1 << 64) - 1 else None


def _expected_rows(input_text: str) -> tuple[tuple[str, str], ...]:
    rows = []
    for line in input_text.splitlines():
        fields = line.split("|")
        if len(fields) < 2 or fields[0] not in {"C", "R", "E"} or not _protocol_token(fields[1]):
            raise ValueError("evaluator input cannot be mapped to one adapter observation")
        rows.append(("EXP" if fields[0] == "E" else "REC", fields[1]))
    if not rows:
        raise ValueError("evaluator input must contain at least one observation case")
    return tuple(rows)


def _parse_adapter(
    stdout: bytes,
    stderr: bytes = b"",
    expected_rows: tuple[tuple[str, str], ...] | None = None,
) -> tuple[bool, dict[str, dict[str, Any]]]:
    """Accept one boot and one canonical primitive row per supplied event."""
    if stderr or len(stdout) > 65536 or not stdout.endswith(b"\n"):
        return False, {}
    try:
        lines = stdout[:-1].decode("ascii").split("\n")
    except UnicodeDecodeError:
        return False, {}
    if len(lines) < 2 or lines[0] != "BOOT|OK" or any(not line for line in lines):
        return False, {}

    observations: dict[str, dict[str, Any]] = {}
    observed_rows: list[tuple[str, str]] = []
    for line in lines[1:]:
        fields = line.split("|")
        if len(fields) not in {9, 12} or fields[0] not in {"REC", "EXP"}:
            return False, {}
        kind, case, status = fields[:3]
        if not _protocol_token(case) or case in observations or status not in {"OK", "ERR"}:
            return False, {}
        if kind == "REC" and len(fields) != 12:
            return False, {}
        if kind == "EXP" and len(fields) != 9:
            return False, {}
        observed_rows.append((kind, case))
        if status == "ERR":
            if any(fields[3:]):
                return False, {}
            observations[case] = {
                "kind": "reconcile" if kind == "REC" else "export",
                "call_ok": False,
            }
            continue

        if kind == "REC":
            state, policy = fields[3:5]
            numeric = [
                _canonical_integer(fields[5]),
                _canonical_integer(fields[6]),
                _canonical_integer(fields[7]),
                _canonical_integer(fields[8], signed=True),
                _canonical_integer(fields[9]),
                _canonical_integer(fields[10]),
            ]
            if (
                state not in {"Applied", "Rejected", "Duplicate"}
                or not _protocol_token(policy)
                or any(value is None for value in numeric)
                or fields[11] not in {"0", "1"}
            ):
                return False, {}
            observations[case] = {
                "kind": "reconcile", "call_ok": True, "state": state,
                "policy": policy, "generation": numeric[0], "horizon": numeric[1],
                "entries": numeric[2], "balance": numeric[3],
                "checkpoint": numeric[4], "pending": numeric[5],
                "recovered": fields[11] == "1",
            }
        else:
            policy = fields[3]
            numeric = [
                _canonical_integer(fields[4]),
                _canonical_integer(fields[5]),
                _canonical_integer(fields[6], signed=True),
                _canonical_integer(fields[7]),
                _canonical_integer(fields[8]),
            ]
            if not _protocol_token(policy) or any(value is None for value in numeric):
                return False, {}
            observations[case] = {
                "kind": "export", "call_ok": True, "policy": policy,
                "generation": numeric[0], "entries": numeric[1],
                "balance": numeric[2], "checkpoint": numeric[3], "pending": numeric[4],
            }
    rows_ok = expected_rows is None or tuple(observed_rows) == expected_rows
    return rows_ok, observations if rows_ok else {}


def _rec(observations: dict[str, dict[str, Any]], case: str, *, state: str,
         policy: str, generation: int, horizon: int, entries: int, balance: int,
         checkpoint: int, pending: int, recovered: bool = False) -> bool:
    return observations.get(case) == {
        "kind": "reconcile", "call_ok": True, "state": state, "policy": policy,
        "generation": generation, "horizon": horizon, "entries": entries,
        "balance": balance, "checkpoint": checkpoint, "pending": pending,
        "recovered": recovered,
    }


def _exp(observations: dict[str, dict[str, Any]], case: str, *, policy: str,
         generation: int, entries: int, balance: int, checkpoint: int, pending: int) -> bool:
    return observations.get(case) == {
        "kind": "export", "call_ok": True, "policy": policy, "generation": generation,
        "entries": entries, "balance": balance, "checkpoint": checkpoint, "pending": pending,
    }


def _run(binary: Path, config: Path, root: Path, input_text: str, output: Path,
         label: str) -> tuple[int, bool, dict[str, dict[str, Any]]]:
    candidate = binary.parent.resolve()
    resolved_binary = binary.resolve()
    resolved_config = config.resolve()
    resolved_root = root.resolve()
    command = [str(resolved_binary), str(resolved_config), str(resolved_root)]
    valid_paths = (
        _regular(binary)
        and _regular(config)
        and resolved_binary.parent == candidate
        and resolved_config.parent == candidate
        and root.parent.resolve() == candidate
        and (_absent(root) or _directory(root))
    )
    if valid_paths:
        root.mkdir(mode=0o700, exist_ok=True)
        resolved_root = root.resolve()
        command[-1] = str(resolved_root)
    static_paths = {
        resolved_binary,
        candidate / "runtime.conf",
        candidate / "src/ledger/reconcile.rs",
        resolved_config,
    }
    static_before = {
        str(path.relative_to(candidate)): _sha(path)
        for path in static_paths
        if _regular(path)
    }
    sandbox = (
        '(version 1)(allow default)(deny network*)(deny process-fork)'
        '(deny file-write* (require-not (subpath ' + json.dumps(str(resolved_root)) + ')))'
        '(deny file-read* (require-all (subpath ' + json.dumps(str(PRIVATE_HOME)) + ') '
        '(require-not (subpath ' + json.dumps(str(binary.parent.resolve())) + '))))'
    )
    if not valid_paths or not _regular(SANDBOX_EXEC):
        reason = (
            "sandbox-exec is unavailable; candidate execution requires the macOS sandbox"
            if not _regular(SANDBOX_EXEC)
            else "candidate inputs are unavailable or unsafe"
        )
        result = subprocess.CompletedProcess(command, 126, b"", reason.encode("utf-8"))
    else:
        try:
            result = subprocess.run(
                [str(SANDBOX_EXEC), "-p", sandbox, *command],
                input=input_text.encode("utf-8"), capture_output=True,
                cwd=binary.parent, timeout=5,
                env={"PATH": "/usr/bin:/bin", "LANG": "C", "TMPDIR": str(root)},
            )
        except subprocess.TimeoutExpired as exc:
            result = subprocess.CompletedProcess(exc.cmd, 124, exc.stdout or b"", exc.stderr or b"")
    (output / f"adapter-{label}.stdout.txt").write_bytes(result.stdout[:65536])
    (output / f"adapter-{label}.stderr.txt").write_bytes(result.stderr[:65536])
    static_after = {
        str(path.relative_to(candidate)): _sha(path)
        for path in static_paths
        if _regular(path)
    }
    protocol_ok, observations = _parse_adapter(
        result.stdout, result.stderr, _expected_rows(input_text)
    )
    accepted = result.returncode == 0 and protocol_ok and static_before == static_after \
        and len(static_before) == len({str(path.relative_to(candidate)) for path in static_paths})
    return result.returncode, accepted, observations if accepted else {}


def _behavior_checks(candidate: Path, output: Path, binary: Path) -> tuple[list[dict[str, Any]], dict[str, int]]:
    static_before = _tree_snapshot(candidate)
    static_shape_ok = static_before is not None and set(static_before) == {
        "src", "src/ledger", "src/ledger/reconcile.rs", "runtime.conf",
        "observation-adapter",
    }
    production = candidate / "runtime-production"
    custom_root = candidate / "runtime-custom"
    custom_config = candidate / "custom.conf"
    custom_config.write_text(CUSTOM_CONFIG, encoding="utf-8")
    config = candidate / "runtime.conf"
    runs: list[tuple[str, int, bool, dict[str, dict[str, Any]]]] = []
    poisoned = False

    def run(label: str, root: Path, input_text: str, selected_config: Path = config):
        nonlocal poisoned
        if poisoned:
            runs.append((label, 126, False, {}))
            return {}
        exit_code, boot, observations = _run(binary, selected_config, root, input_text, output, label)
        runs.append((label, exit_code, boot, observations))
        if exit_code != 0 or not boot:
            poisoned = True
        return observations

    phase1 = run("pending", production, PHASE_PENDING)
    ledger = production / "main" / "ledger.log"
    phase1_journal = _read_regular(ledger)
    phase1_checkpoint = _read_regular(production / "main" / "checkpoint.state")
    phase1_tree = _tree_snapshot(production)

    phase2 = run("delayed", production, PHASE_DELAYED)
    phase2_journal = _read_regular(ledger)
    phase2_tree = _tree_snapshot(production)
    marker = production / "main" / "reconcile.marker"
    if phase2_tree is not None and _directory(production) and _directory(marker.parent):
        marker.write_bytes(b"stale-before-reload\n")
        pre_reload_tree = _tree_snapshot(production)
    else:
        poisoned = True
        pre_reload_tree = None
    phase3 = run("reload", production, PHASE_RELOAD)
    phase3_journal = _read_regular(ledger)
    phase3_tree = _tree_snapshot(production)
    phase4 = run("direct", production, PHASE_DIRECT)
    phase4_tree = _tree_snapshot(production)
    shuffled = run("shuffled", production, PHASE_SHUFFLED)
    shuffled_tree = _tree_snapshot(production)
    negative = run("negative", production, PHASE_NEGATIVE)
    negative_tree = _tree_snapshot(production)
    custom = run("custom", custom_root, CUSTOM_INPUT, custom_config)
    custom_tree = _tree_snapshot(custom_root)

    expected_cases = {
        "pending", "delayed", "retry-charge", "charge-two", "reverse-two", "retry-reverse",
        "export-main", "shuffle-b", "shuffle-pending", "shuffle-a", "far-charge",
        "outside-horizon", "wrong-schema", "custom-pending", "custom-charge", "custom-export",
    }
    expected_phase1_journal = b"P|rev-late|104|charge-late|100\n"
    expected_phase2 = (expected_phase1_journal
                       + b"C|charge-late|100|2500\n"
                       + b"T|rev-late|charge-late|2500\n")
    expected_phase3 = expected_phase2 + b"C|charge-two|105|700\n"
    expected_phase4 = expected_phase3 + b"R|rev-two|109|charge-two|105|700\n"
    shuffled_exact = (b"C|charge-b|202|300\n"
                      b"P|rev-a|204|charge-a|200\n"
                      b"C|charge-a|200|500\n"
                      b"T|rev-a|charge-a|500\n")
    custom_exact = (b"P|custom-rev|12|custom-charge|10\n"
                    b"C|custom-charge|10|80\n"
                    b"T|custom-rev|custom-charge|80\n")
    phase1_files = {
        "main/ledger.log": expected_phase1_journal,
        "main/checkpoint.state": b"99\n",
        "main/balance.state": b"0\n",
    }
    phase2_files = {
        "main/ledger.log": expected_phase2,
        "main/checkpoint.state": b"104\n",
        "main/balance.state": b"0\n",
    }
    phase3_files = {
        "main/ledger.log": expected_phase3,
        "main/checkpoint.state": b"105\n",
        "main/balance.state": b"700\n",
    }
    phase4_files = {
        "main/ledger.log": expected_phase4,
        "main/checkpoint.state": b"109\n",
        "main/balance.state": b"0\n",
        "export-main.state": b"ledger-export|5|0|109|0\n",
    }
    shuffled_files = {
        **phase4_files,
        "shuffled/ledger.log": shuffled_exact,
        "shuffled/checkpoint.state": b"204\n",
        "shuffled/balance.state": b"300\n",
    }
    final_production_files = {
        **shuffled_files,
        "negative/ledger.log": b"C|charge-far|300|50\n",
        "negative/checkpoint.state": b"300\n",
        "negative/balance.state": b"50\n",
    }
    final_custom_files = {
        "custom/ledger.log": custom_exact,
        "custom/checkpoint.state": b"12\n",
        "custom/balance.state": b"0\n",
        "custom-export.state": b"ledger-export|3|0|12|0\n",
    }
    expected_phase1_tree = _expected_tree(phase1_files)
    expected_phase2_tree = _expected_tree(phase2_files)
    expected_pre_reload_tree = _expected_tree({
        **phase2_files, "main/reconcile.marker": b"stale-before-reload\n"
    })
    expected_phase3_tree = _expected_tree(phase3_files)
    expected_phase4_tree = _expected_tree(phase4_files)
    expected_shuffled_tree = _expected_tree(shuffled_files)
    expected_negative_tree = _expected_tree(final_production_files)
    expected_custom_tree = _expected_tree(final_custom_files)
    merged: dict[str, dict[str, Any]] = {}
    for _, _, _, observations in runs:
        for case, value in observations.items():
            if case in merged:
                merged[case] = {"duplicate_across_runs": True}
            else:
                merged[case] = value
    protocol_ok = all(exit_code == 0 and boot for _, exit_code, boot, _ in runs) \
        and set(merged) == expected_cases
    checks = [_check("adapter_protocol_complete", protocol_ok,
                     "generic adapter emitted primitive rows; expected comparisons stayed in Python")]

    checks.append(_check("pending_reversal_pins_checkpoint",
        _rec(phase1, "pending", state="Applied", policy="CYPRESS", generation=7, horizon=4,
             entries=1, balance=0, checkpoint=99, pending=1)
        and phase1_journal == expected_phase1_journal
        and phase1_checkpoint == b"99\n"
        and phase1_tree == expected_phase1_tree))
    checks.append(_check("delayed_charge_resolves_exactly_once",
        _rec(phase2, "delayed", state="Applied", policy="CYPRESS", generation=7, horizon=4,
             entries=3, balance=0, checkpoint=104, pending=0)))
    checks.append(_check("journal_is_append_only_with_resolution_tombstone",
        phase2_journal == expected_phase2
        and isinstance(phase1_journal, bytes)
        and phase2_journal.startswith(phase1_journal)
        and phase2_tree == expected_phase2_tree))
    checks.append(_check("stale_marker_reload_and_duplicate_suppression",
        _rec(phase3, "retry-charge", state="Duplicate", policy="CYPRESS", generation=7,
             horizon=4, entries=3, balance=0, checkpoint=104, pending=0, recovered=True)
        and _rec(phase3, "charge-two", state="Applied", policy="CYPRESS", generation=7,
                 horizon=4, entries=4, balance=700, checkpoint=105, pending=0)
        and isinstance(phase2_journal, bytes)
        and isinstance(phase3_journal, bytes)
        and phase3_journal.startswith(phase2_journal)
        and pre_reload_tree == expected_pre_reload_tree
        and phase3_tree == expected_phase3_tree
        and _absent(marker)))
    checks.append(_check("direct_reversal_and_retry_are_exact",
        _rec(phase4, "reverse-two", state="Applied", policy="CYPRESS", generation=7,
             horizon=4, entries=5, balance=0, checkpoint=109, pending=0)
        and _rec(phase4, "retry-reverse", state="Duplicate", policy="CYPRESS", generation=7,
                 horizon=4, entries=5, balance=0, checkpoint=109, pending=0)
        and phase4_tree == expected_phase4_tree))
    negative_ledger = production / "negative" / "ledger.log"
    checks.append(_check("horizon_and_schema_rejections_preserve_journal",
        _rec(negative, "far-charge", state="Applied", policy="CYPRESS", generation=7,
             horizon=4, entries=1, balance=50, checkpoint=300, pending=0)
        and _rec(negative, "outside-horizon", state="Rejected", policy="CYPRESS", generation=7,
                 horizon=4, entries=1, balance=50, checkpoint=300, pending=0)
        and _rec(negative, "wrong-schema", state="Rejected", policy="CYPRESS", generation=7,
                 horizon=4, entries=1, balance=50, checkpoint=300, pending=0)
        and _read_regular(negative_ledger) == b"C|charge-far|300|50\n"
        and negative_tree == expected_negative_tree))

    shuffled_ledger = production / "shuffled" / "ledger.log"
    checks.append(_check("shuffled_events_preserve_all_ledger_effects",
        _rec(shuffled, "shuffle-b", state="Applied", policy="CYPRESS", generation=7,
             horizon=4, entries=1, balance=300, checkpoint=202, pending=0)
        and _rec(shuffled, "shuffle-pending", state="Applied", policy="CYPRESS", generation=7,
                 horizon=4, entries=2, balance=300, checkpoint=199, pending=1)
        and _rec(shuffled, "shuffle-a", state="Applied", policy="CYPRESS", generation=7,
                 horizon=4, entries=4, balance=300, checkpoint=204, pending=0)
        and _read_regular(shuffled_ledger) == shuffled_exact
        and shuffled_tree == expected_shuffled_tree))

    export_path = production / "export-main.state"
    checks.append(_check("unrelated_export_behavior_is_unchanged",
        _exp(phase4, "export-main", policy="LARCH", generation=3, entries=5,
             balance=0, checkpoint=109, pending=0)
        and _read_regular(export_path) == b"ledger-export|5|0|109|0\n"
        and phase4_tree == expected_phase4_tree))

    custom_ledger = custom_root / "custom" / "ledger.log"
    checks.append(_check("alternate_policy_drives_behavior_without_hardcoding",
        _rec(custom, "custom-pending", state="Applied", policy="MAPLE", generation=2,
             horizon=2, entries=1, balance=0, checkpoint=9, pending=1)
        and _rec(custom, "custom-charge", state="Applied", policy="MAPLE", generation=2,
                 horizon=2, entries=3, balance=0, checkpoint=12, pending=0)
        and _exp(custom, "custom-export", policy="OAK", generation=1, entries=3,
                 balance=0, checkpoint=12, pending=0)
        and _read_regular(custom_ledger) == custom_exact
        and _read_regular(custom_root / "custom-export.state") == b"ledger-export|3|0|12|0\n"
        and custom_tree == expected_custom_tree))

    all_journals = [ledger, shuffled_ledger, negative_ledger, custom_ledger]
    expected_candidate = dict(static_before or {})
    expected_candidate["custom.conf"] = ("file", CUSTOM_CONFIG.encode("utf-8"))
    expected_candidate.update(_prefix_tree(expected_negative_tree, "runtime-production"))
    expected_candidate.update(_prefix_tree(expected_custom_tree, "runtime-custom"))
    candidate_exact = static_shape_ok and _tree_snapshot(candidate) == expected_candidate
    checks.append(_check("real_state_files_and_atomic_cleanup",
        all(_regular(path) for path in all_journals)
        and _read_regular(production / "main" / "balance.state") == b"0\n"
        and _read_regular(production / "main" / "checkpoint.state") == b"109\n"
        and negative_tree == expected_negative_tree
        and custom_tree == expected_custom_tree
        and candidate_exact))
    production_reconciliation_cases = {
        case for case in expected_cases
        if not case.startswith("custom-") and case != "export-main"
    }
    checks.append(_check("approved_reconciliation_owner_is_exact",
        production_reconciliation_cases <= set(merged)
        and all(merged[case].get("policy") == "CYPRESS"
                and merged[case].get("generation") == 7
                for case in production_reconciliation_cases)))
    return checks, {label: exit_code for label, exit_code, _, _ in runs}


def _finish(output: Path, checks: list[dict[str, Any]], **extra: Any) -> dict[str, Any]:
    score = sum(1 for item in checks if item["pass"])
    report: dict[str, Any] = {"checks": checks, "score": score, "total": len(checks),
                              "all_pass": score == len(checks)}
    report.update(extra)
    (output / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                                         encoding="utf-8")
    return report


def evaluate(workspace: Path, output: Path) -> dict[str, Any]:
    workspace = Path(workspace).resolve()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    checks: list[dict[str, Any]] = []
    shape_errors = _workspace_shape_errors(workspace)
    application_bytes = {
        name: _read_regular(workspace / name) for name in APPLICATION_FILES
    }
    app_ok = all(content is not None for content in application_bytes.values()) \
        and not shape_errors
    application_detail = "regular application files and exact workspace shape"
    if shape_errors:
        application_detail = "unexpected or unsafe workspace paths: " + ", ".join(shape_errors)
    elif not app_ok:
        application_detail = "application file missing, unsafe, or oversized"
    checks.append(_check("application_files_are_regular", app_ok, application_detail))

    differences = []
    for name in PROTECTED_FILES:
        candidate = workspace / name
        canonical = FIXTURE / name
        same = (_regular(candidate)
                and candidate.stat().st_size == canonical.stat().st_size
                and _sha(candidate) == _sha(canonical)
                and stat.S_IMODE(candidate.stat().st_mode) == stat.S_IMODE(canonical.stat().st_mode))
        if not same:
            differences.append(name)
    protected_ok = not differences
    checks.append(_check("protected_files_intact", protected_ok,
                         "changed: " + ", ".join(differences) if differences else "all protected bytes and modes match"))
    changed_files = [
        name for name, content in application_bytes.items()
        if content is not None and content != _read_regular(FIXTURE / name)
    ]
    checks.extend(_decision_checks(workspace, changed_files))

    candidate = output / "candidate"
    candidate.mkdir(mode=0o700)
    compile_result: subprocess.CompletedProcess[bytes] | None = None
    compile_tree: dict[str, tuple[str, bytes | None]] | None = None
    if app_ok and protected_ok:
        try:
            for name in APPLICATION_FILES:
                destination = candidate / name
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(workspace / name, destination)
            adapter = candidate / "observation_adapter.rs"
            shutil.copyfile(FIXTURE / "observation_adapter.rs", adapter)
        except OSError as exc:
            compile_result = subprocess.CompletedProcess(
                [str(RUSTC) if RUSTC else "rustc"], 126, b"",
                f"candidate preparation failed: {exc}".encode("utf-8")
            )
            adapter = candidate / "observation_adapter.rs"
        binary = candidate / "observation-adapter"
        rust_command: list[str] = []
        compile_profile = ""
        if RUSTC is not None:
            toolchain_root = RUSTC.parent.parent.resolve()
            rust_command = [str(RUSTC), "--edition=2021", "-C", "opt-level=0", "-C",
                            "debuginfo=0", adapter.name, "-o", str(binary)]
            compile_profile = (
                '(version 1)(allow default)(deny network*)'
                '(deny file-write* (require-not (subpath ' + json.dumps(str(candidate.resolve())) + ')))'
                '(deny file-read* (require-all (subpath ' + json.dumps(str(PRIVATE_HOME)) + ') '
                '(require-not (subpath ' + json.dumps(str(candidate.resolve())) + ')) '
                '(require-not (subpath ' + json.dumps(str(toolchain_root)) + '))))'
            )
        if compile_result is None and _regular(RUSTC) and _regular(SANDBOX_EXEC):
            try:
                compile_result = subprocess.run([str(SANDBOX_EXEC), "-p", compile_profile,
                                                 *rust_command],
                                                cwd=candidate, capture_output=True,
                                                timeout=30, env={"PATH": "/usr/bin:/bin",
                                                                "LANG": "C", "TMPDIR": str(candidate)})
            except subprocess.TimeoutExpired as exc:
                compile_result = subprocess.CompletedProcess(exc.cmd, 124, exc.stdout or b"", exc.stderr or b"")
        elif compile_result is None:
            if not _regular(RUSTC):
                reason = "rustc is unavailable; set RUSTC to an executable compiler path"
            elif sys.platform != "darwin":
                reason = (
                    f"unsupported platform {sys.platform!r}; candidate compilation requires "
                    "the macOS sandbox-exec sandbox"
                )
            else:
                reason = (
                    "sandbox-exec is unavailable; candidate compilation requires the macOS sandbox"
                )
            compile_result = subprocess.CompletedProcess(rust_command, 126, b"", reason.encode("utf-8"))
        adapter.unlink(missing_ok=True)
        compile_tree = _tree_snapshot(candidate)

    expected_static_paths = {
        "src", "src/ledger", "src/ledger/reconcile.rs", "runtime.conf",
        "observation-adapter",
    }
    copied_inputs_ok = compile_tree is not None and all(
        compile_tree.get(name) == ("file", content)
        for name, content in application_bytes.items()
        if content is not None
    )
    binary_entry = compile_tree.get("observation-adapter") if compile_tree is not None else None
    compile_shape_ok = (
        compile_tree is not None
        and set(compile_tree) == expected_static_paths
        and copied_inputs_ok
        and binary_entry is not None
        and binary_entry[0] == "file"
        and bool(binary_entry[1])
    )
    compile_ok = (compile_result is not None and compile_result.returncode == 0
                  and compile_shape_ok)
    (output / "compile.stderr.txt").write_bytes(
        (compile_result.stderr if compile_result else
         b"sandbox, rustc, protected files, or workspace inputs unavailable")[:65536]
    )
    if compile_result:
        (output / "compile.stdout.txt").write_bytes(compile_result.stdout[:65536])
    checks.append(_check(
        "rust_compiles", compile_ok,
        f"rustc={RUSTC}; sandbox={SANDBOX_EXEC}; "
        f"exit={compile_result.returncode if compile_result else 'not-run'}; "
        f"static_tree_exact={compile_shape_ok}",
    ))
    exits: dict[str, int] = {}
    behavior_names = (
        "adapter_protocol_complete", "pending_reversal_pins_checkpoint",
        "delayed_charge_resolves_exactly_once", "journal_is_append_only_with_resolution_tombstone",
        "stale_marker_reload_and_duplicate_suppression", "direct_reversal_and_retry_are_exact",
        "horizon_and_schema_rejections_preserve_journal", "shuffled_events_preserve_all_ledger_effects",
        "unrelated_export_behavior_is_unchanged", "alternate_policy_drives_behavior_without_hardcoding",
        "real_state_files_and_atomic_cleanup", "approved_reconciliation_owner_is_exact",
    )
    if compile_ok:
        behavior, exits = _behavior_checks(candidate, output, candidate / "observation-adapter")
        checks.extend(behavior)
    else:
        checks.extend(_check(name, False, "candidate did not compile") for name in behavior_names)

    version = subprocess.run([str(RUSTC), "--version"], capture_output=True, text=True).stdout.strip() \
        if _regular(RUSTC) else None
    return _finish(output, checks, rustc=str(RUSTC), rustc_version=version,
                   compile_exit=compile_result.returncode if compile_result else None,
                   adapter_exits=exits, changed_application_files=changed_files,
                   oracle_boundary="generic primitive adapter; Python owns expected rows and durable-byte comparisons",
                   recovery_claim="separate process reload after a pre-existing marker; no claim of arbitrary crash atomicity")


def _reference_decision(changed_files: list[str]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "cause": "A valid ordered-batch owner remained routed after schema, wire, owner, and pending-checkpoint obligations changed.",
        "evidence_ids": ["schema-v3-rollover", "wire-v2-transition", "owner-alias-migration",
                         "epoch7-approved", "obligation-reconcile-path", "incident-current"],
        "causal_path": [
            {"from": "schema-v3-rollover", "to": "wire-v2-transition", "relation": "encoded_by"},
            {"from": "wire-v2-transition", "to": "owner-alias-migration", "relation": "assigned_owner_by"},
            {"from": "owner-alias-migration", "to": "epoch7-approved", "relation": "promoted_as"},
            {"from": "epoch7-approved", "to": "obligation-reconcile-path", "relation": "guarded_by"},
            {"from": "obligation-reconcile-path", "to": "incident-current", "relation": "applies_when_path_edited"},
        ],
        "rejected_hypotheses": [
            {"hypothesis": "Apply the exact-word performance runbook to the ledger.",
             "evidence_ids": ["performance-runbook-decoy", "performance-runbook-validation"],
             "reason": "Its validation is limited to disposable analytics summaries."},
            {"hypothesis": "Select the numerically newer generation-eight fork.",
             "evidence_ids": ["fork-generation8-proposal", "fork-generation8-rejected"],
             "reason": "The fork was rejected without crash/reload compatibility proof."},
        ],
        "changed_files": changed_files,
    }


def _write_decision(workspace: Path, changed_files: list[str]) -> None:
    (workspace / DECISION_FILE).write_text(json.dumps(_reference_decision(changed_files), indent=2) + "\n",
                                           encoding="utf-8")


def _replace(path: Path, old: str, new: str) -> None:
    text = path.read_text(encoding="utf-8")
    if text.count(old) != 1:
        raise AssertionError(f"control anchor count for {old!r} was {text.count(old)}")
    path.write_text(text.replace(old, new), encoding="utf-8")


def preflight(output: Path) -> dict[str, Any]:
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    workspaces = output / "workspaces"
    workspaces.mkdir()
    variants = ("baseline", "reference", "batch-size-runbook", "horizon-bump",
                "cedar-pin", "newer-fork", "export-owner", "blanket-owner",
                "remove-tombstone")
    reports: dict[str, dict[str, Any]] = {}
    for variant in variants:
        workspace = workspaces / variant
        shutil.copytree(FIXTURE, workspace)
        config = workspace / "runtime.conf"
        source = workspace / "src/ledger/reconcile.rs"
        if variant == "reference":
            _replace(config, "route.RECONCILE=CEDAR@7", "route.RECONCILE=CYPRESS@7")
        elif variant == "batch-size-runbook":
            _replace(config, "diagnostic.batch_size=32", "diagnostic.batch_size=128")
        elif variant == "horizon-bump":
            _replace(config, "policy.CEDAR.REVERSAL_HORIZON=4", "policy.CEDAR.REVERSAL_HORIZON=128")
        elif variant == "cedar-pin":
            _replace(config, "policy.CEDAR.checkpoint_mode=MAX_SEEN", "policy.CEDAR.checkpoint_mode=PIN_PENDING")
        elif variant == "newer-fork":
            _replace(config, "route.RECONCILE=CEDAR@7", "route.RECONCILE=FALCON@8")
        elif variant == "export-owner":
            _replace(config, "route.RECONCILE=CEDAR@7", "route.RECONCILE=LARCH@3")
        elif variant == "blanket-owner":
            _replace(config, "route.RECONCILE=CEDAR@7", "route.RECONCILE=CYPRESS@7")
            _replace(config, "route.EXPORT_SNAPSHOT=LARCH@3", "route.EXPORT_SNAPSHOT=CYPRESS@7")
        elif variant == "remove-tombstone":
            _replace(config, "route.RECONCILE=CEDAR@7", "route.RECONCILE=CYPRESS@7")
            _replace(config, "diagnostic.batch_size=32", "diagnostic.batch_size=128")
            _replace(source, 'lines.push(format!("T|{}|{}|{}", item.reversal_id, event.id, amount_cents));', "")
        changed = [name for name in APPLICATION_FILES
                   if (workspace / name).read_bytes() != (FIXTURE / name).read_bytes()]
        if changed:
            _write_decision(workspace, changed)
        reports[variant] = evaluate(workspace, output / f"evaluated-{variant}")

    flags = lambda report: {item["name"]: item["pass"] for item in report["checks"]}
    baseline = flags(reports["baseline"])
    reference = flags(reports["reference"])
    blanket = flags(reports["blanket-owner"])
    controls = [
        _check("baseline_control_executes", baseline["rust_compiles"]
               and baseline["adapter_protocol_complete"]),
        _check("reference_is_accepted", reports["reference"]["all_pass"]),
        _check("batch_size_runbook_patch_is_rejected", not reports["batch-size-runbook"]["all_pass"]),
        _check("horizon_bump_is_rejected", not reports["horizon-bump"]["all_pass"]),
        _check("same_horizon_cedar_pin_is_rejected", not reports["cedar-pin"]["all_pass"]
               and flags(reports["cedar-pin"])["journal_is_append_only_with_resolution_tombstone"]
               and not flags(reports["cedar-pin"])["approved_reconciliation_owner_is_exact"]),
        _check("newer_fork_is_rejected", not reports["newer-fork"]["all_pass"]),
        _check("export_owner_transfer_is_rejected", not reports["export-owner"]["all_pass"]),
        _check("blanket_owner_change_is_rejected", not reports["blanket-owner"]["all_pass"]
               and blanket["delayed_charge_resolves_exactly_once"]
               and not blanket["unrelated_export_behavior_is_unchanged"]),
        _check("runbook_tombstone_removal_is_rejected", not reports["remove-tombstone"]["all_pass"]
               and not flags(reports["remove-tombstone"])["journal_is_append_only_with_resolution_tombstone"]),
    ]
    score = sum(1 for item in controls if item["pass"])
    report = {"checks": controls, "score": score, "total": len(controls),
              "all_pass": score == len(controls), "controls": reports,
              "fixture": str(FIXTURE), "rustc": str(RUSTC),
              "oracle_boundary": "generic adapter emits primitives; Python owns expectations and reads durable files"}
    (output / "preflight-report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                                                   encoding="utf-8")
    return report
