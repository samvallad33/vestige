#!/usr/bin/env python3
"""Recheck recorded Demo 4 application files without running any model."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import stat
import sys
import tempfile
from pathlib import Path
from typing import Any, Sequence


HERE = Path(__file__).resolve().parent
HARNESS = HERE / "harness"
sys.path.insert(0, str(HARNESS))

import evaluator  # noqa: E402


ARMS = ("control", "mcp-memory-service", "vestige")
DEFAULT_EVIDENCE = HERE / "evidence"
RESULT_FILE = "application-recheck.json"
HISTORICAL_RUN_BOUNDARY = {
    "original_run": "context-intention-1ux5s5ji",
    "control": {"model_finished_naturally": True, "timeout": False},
    "mcp-memory-service": {
        "model_finished_naturally": False,
        "timeout": True,
        "timeout_seconds": 900,
        "final_application_files_passed": True,
    },
    "vestige": {"model_finished_naturally": True, "timeout": False},
    "boundary": (
        "This application-only recheck does not rerun or complete the historical model "
        "sessions. The Memory Service session timed out after 900 seconds even though its "
        "recorded final application files pass the evaluator."
    ),
}


def _update_hash(digest: Any, value: str) -> None:
    encoded = value.encode("utf-8")
    digest.update(len(encoded).to_bytes(8, "big"))
    digest.update(encoded)


def _tree_fingerprint(root: Path) -> str:
    """Hash a regular tree, rejecting links and special files."""
    try:
        root_mode = root.lstat().st_mode
    except OSError as exc:
        raise ValueError(f"project directory is unavailable: {root}") from exc
    if stat.S_ISLNK(root_mode) or not stat.S_ISDIR(root_mode):
        raise ValueError(f"project must be a real directory: {root}")
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix()):
        relative = path.relative_to(root).as_posix()
        mode = path.lstat().st_mode
        if stat.S_ISDIR(mode):
            kind = "directory"
        elif stat.S_ISREG(mode):
            kind = "file"
        elif stat.S_ISLNK(mode):
            raise ValueError(f"project tree contains a symlink: {relative}")
        else:
            raise ValueError(f"project tree contains a special file: {relative}")
        _update_hash(digest, relative)
        _update_hash(digest, kind)
        _update_hash(digest, f"{stat.S_IMODE(mode):04o}")
        if kind == "file":
            _update_hash(digest, str(path.stat().st_size))
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
    return digest.hexdigest()


def _absolute_without_resolving(path: Path) -> Path:
    path = path.expanduser()
    return path if path.is_absolute() else Path.cwd() / path


def _reject_symlinked_path(path: Path) -> None:
    """Reject a symlink in any existing component of an input path."""
    absolute = _absolute_without_resolving(path)
    components = [absolute, *absolute.parents]
    for component in reversed(components):
        try:
            mode = component.lstat().st_mode
        except FileNotFoundError:
            continue
        except OSError as exc:
            raise ValueError(f"input path component is unavailable: {component}") from exc
        if stat.S_ISLNK(mode):
            raise ValueError(f"input path contains a symlinked component: {component}")


def _regular_file_identity(path: Path) -> dict[str, Any] | None:
    try:
        mode = path.lstat().st_mode
    except OSError:
        return None
    if path.is_symlink() or not stat.S_ISREG(mode):
        return None
    return {
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "mode": f"{stat.S_IMODE(mode):04o}",
        "size": path.stat().st_size,
    }


def _validate_recorded_protected_inputs(
    evidence: Path, projects: dict[str, Path]
) -> dict[str, dict[str, Any]]:
    inputs = evidence / "inputs"
    expected: dict[str, dict[str, Any]] = {}
    for name in evaluator.PROTECTED_FILES:
        stored_identity = _regular_file_identity(inputs / name)
        canonical_identity = _regular_file_identity(evaluator.FIXTURE / name)
        if stored_identity is None:
            raise ValueError(f"recorded protected input is missing or unsafe: inputs/{name}")
        if canonical_identity is None:
            raise ValueError(f"canonical protected fixture is missing or unsafe: fixture/{name}")
        expected[name] = {
            "sha256": stored_identity["sha256"],
            "mode": canonical_identity["mode"],
            "size": stored_identity["size"],
            "evidence_input_storage_mode": stored_identity["mode"],
        }

    differences = []
    for arm, project in projects.items():
        for name, identity in expected.items():
            candidate_identity = _regular_file_identity(project / name)
            recorded_identity = {
                key: identity[key] for key in ("sha256", "mode", "size")
            }
            if candidate_identity != recorded_identity:
                differences.append(f"{arm}/project/{name}")
    if differences:
        raise ValueError(
            "recorded protected inputs differ from evidence/inputs: " + ", ".join(differences)
        )
    return expected


def _prepare_output(output: Path) -> None:
    if output.is_symlink():
        raise ValueError(f"--output must not be a symlink: {output}")
    if output.exists():
        if not output.is_dir():
            raise ValueError(f"--output is not a directory: {output}")
        if any(output.iterdir()):
            raise ValueError(f"--output must be empty: {output}")
    else:
        output.mkdir(parents=True, exist_ok=False)


def run_recheck(evidence: Path, output: Path) -> dict[str, Any]:
    """Evaluate all three recorded projects through disposable copies."""
    requested_evidence = _absolute_without_resolving(Path(evidence))
    _reject_symlinked_path(requested_evidence)
    try:
        evidence = requested_evidence.resolve(strict=True)
    except OSError as exc:
        raise ValueError(f"recorded evidence directory is unavailable: {requested_evidence}") from exc
    requested_output = _absolute_without_resolving(Path(output))
    if requested_output.is_symlink():
        raise ValueError(f"--output must not be a symlink: {requested_output}")
    output = requested_output.resolve()
    if output == evidence or output.is_relative_to(evidence):
        raise ValueError("--output must be outside the recorded evidence directory")
    projects: dict[str, Path] = {}
    for arm in ARMS:
        requested_project = evidence / arm / "project"
        _reject_symlinked_path(requested_project)
        try:
            project = requested_project.resolve(strict=True)
        except OSError as exc:
            raise ValueError(f"recorded project is unavailable: {arm}/project") from exc
        if not project.is_relative_to(evidence):
            raise ValueError(f"recorded project escapes the evidence directory: {arm}/project")
        projects[arm] = project
    source_before = {arm: _tree_fingerprint(project) for arm, project in projects.items()}
    protected_identities = _validate_recorded_protected_inputs(evidence, projects)
    _prepare_output(output)

    arm_results: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="before-you-change-that-recheck-") as temporary:
        temporary_root = Path(temporary)
        for arm in ARMS:
            source = projects[arm]
            disposable = temporary_root / arm
            shutil.copytree(source, disposable, symlinks=False, copy_function=shutil.copy2)
            copied_before = _tree_fingerprint(disposable)
            recorded_launcher = _regular_file_identity(disposable / "reproduce.sh")
            shutil.copy2(evaluator.FIXTURE / "reproduce.sh", disposable / "reproduce.sh")
            portable_launcher = _regular_file_identity(disposable / "reproduce.sh")
            if recorded_launcher is None or portable_launcher is None:
                raise ValueError(f"unable to adapt disposable launcher for {arm}")
            application_files = {
                name: _regular_file_identity(source / name)
                for name in evaluator.APPLICATION_FILES
            }
            application_copy_exact = all(
                identity is not None
                and _regular_file_identity(disposable / name) == identity
                for name, identity in application_files.items()
            )
            adapted_before_evaluation = _tree_fingerprint(disposable)
            evaluation_output = output / arm
            error: dict[str, str] | None = None
            report: dict[str, Any] | None = None
            try:
                report = evaluator.evaluate(disposable, evaluation_output)
            except Exception as exc:  # Preserve a machine-readable fail-closed result.
                error = {"type": type(exc).__name__, "message": str(exc)}
            copied_after = _tree_fingerprint(disposable)
            source_after = _tree_fingerprint(source)
            arm_result: dict[str, Any] = {
                "arm": arm,
                "source_from_evidence_root": f"{arm}/project",
                "source_tree_sha256": source_before[arm],
                "application_files": application_files,
                "application_files_unchanged_by_adaptation": application_copy_exact,
                "copy_fidelity": copied_before == source_before[arm],
                "recorded_protected_inputs_match": True,
                "controlled_disposable_adaptations": [
                    {
                        "path": "reproduce.sh",
                        "recorded": recorded_launcher,
                        "portable": portable_launcher,
                        "reason": (
                            "The recorded public launcher retains its manifest-bound path alias. "
                            "Only this disposable copy receives the portable RUSTC/PATH launcher "
                            "required by the canonical evaluator fixture."
                        ),
                    }
                ],
                "disposable_input_unchanged_after_adaptation": (
                    copied_after == adapted_before_evaluation
                ),
                "recorded_evidence_unchanged": source_after == source_before[arm],
                "application_checks_pass": bool(report and report.get("all_pass")),
                "score": report.get("score") if report else None,
                "total": report.get("total") if report else None,
                "report": f"{arm}/report.json" if report else None,
            }
            if error is not None:
                arm_result["error"] = error
            arm_results.append(arm_result)

    all_pass = all(
        arm["application_checks_pass"]
        and arm["application_files_unchanged_by_adaptation"]
        and arm["copy_fidelity"]
        and arm["recorded_protected_inputs_match"]
        and arm["disposable_input_unchanged_after_adaptation"]
        and arm["recorded_evidence_unchanged"]
        for arm in arm_results
    )
    result: dict[str, Any] = {
        "schema": "vestige.before-you-change-that.application-recheck.v1",
        "recheck_kind": "recorded-final-application-only",
        "model_execution_performed": False,
        "network_access_required": False,
        "workspace_mode": "disposable-copies",
        "controlled_adaptation_boundary": (
            "After each recorded project is verified against evidence/inputs and copied, "
            "only disposable reproduce.sh bytes and mode are replaced by the portable "
            "canonical fixture launcher. Recorded evidence and final application source/config "
            "are not modified."
        ),
        "recorded_protected_inputs": {
            "validated_against": "inputs/",
            "files": protected_identities,
        },
        "evaluator": {
            "checks_per_arm": 21,
            "oracle_boundary": (
                "The generic Rust adapter emits primitive observations; Python owns the "
                "expected rows and durable-byte comparisons."
            ),
            "sandbox_boundary": (
                "Candidate compilation and execution require macOS sandbox-exec and fail "
                "closed when that sandbox is unavailable."
            ),
        },
        "arms": arm_results,
        "all_application_checks_pass": all_pass,
        "historical_model_run": HISTORICAL_RUN_BOUNDARY,
    }
    (output / RESULT_FILE).write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--evidence",
        type=Path,
        default=DEFAULT_EVIDENCE,
        help="recorded evidence directory (default: benchmark evidence directory)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="new or empty directory for recheck artifacts",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        result = run_recheck(arguments.evidence, arguments.output)
    except (OSError, ValueError) as exc:
        print(f"application recheck refused: {exc}", file=sys.stderr)
        return 2

    for arm in result["arms"]:
        status = "PASS" if arm["application_checks_pass"] else "FAIL"
        print(f"{arm['arm']}: {status} ({arm['score']}/{arm['total']})")
    print(f"No model was run. Combined report: {arguments.output / RESULT_FILE}")
    return 0 if result["all_application_checks_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
