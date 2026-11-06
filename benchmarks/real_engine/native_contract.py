"""Lossless MCP catalog adaptation and worker-local host-effect boundaries.

This module does not retrieve or transform memory. MCP's initialize.instructions
has no separate Chat Completions tool field, so its exact bytes are appended once
to memory_status's description. Native descriptions and input schemas otherwise
remain unchanged; system/user prompts are not modified.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
from typing import Any


INSTRUCTIONS_CARRIER = "memory_status.description"
INSTRUCTIONS_SEPARATOR = "\n\nMCP server instructions:\n"


class NativeBoundaryViolation(RuntimeError):
    """A host boundary refused an effect before a native call was attempted."""


def digest_json(value: Any) -> str:
    text = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(text.encode()).hexdigest()


def model_tools(catalog: list[dict], initialize: dict) -> list[dict]:
    """Advertise every native tool, with its exact JSON schema and description."""
    instructions = initialize.get("instructions")
    if not isinstance(instructions, str) or not instructions:
        raise RuntimeError("native initialize.instructions is required")
    names = [tool.get("name") for tool in catalog]
    if not names or any(not isinstance(name, str) or not name for name in names):
        raise RuntimeError("native tool catalog has a missing name")
    if len(set(names)) != len(names) or "memory_status" not in names:
        raise RuntimeError("native tool catalog is duplicated or lacks discovery")
    result = []
    for tool in catalog:
        description = tool.get("description", "")
        schema = tool.get("inputSchema")
        if not isinstance(description, str) or not isinstance(schema, dict):
            raise RuntimeError("native tool description/schema is malformed")
        if tool["name"] == "memory_status":
            description += INSTRUCTIONS_SEPARATOR + instructions
        result.append({"type": "function", "function": {
            "name": tool["name"], "description": description,
            "parameters": deepcopy(schema)}})
    return result


def contract_identity(catalog: list[dict], initialize: dict) -> dict:
    adapted = model_tools(catalog, initialize)
    return {"complete_catalog": True, "tool_count": len(catalog),
            "catalog_sha256": digest_json(catalog),
            "initialize_sha256": digest_json(initialize),
            "model_tools_sha256": digest_json(adapted),
            "instructions_sha256": hashlib.sha256(initialize["instructions"].encode()).hexdigest(),
            "instructions_carrier": INSTRUCTIONS_CARRIER}


def validate_worker_arguments(name: str, arguments: dict, *, roots: tuple[Path, ...],
                              cwd: Path, home: Path) -> None:
    """Keep the complete catalog visible while refusing ambient host effects.

    A refusal is a harness boundary event, never a fabricated native tool result.
    Ordinary memory contents/queries are opaque: only known filesystem argument
    fields are checked. No keywords or benchmark data are inspected.
    """
    if not isinstance(arguments, dict):
        raise NativeBoundaryViolation("native tool arguments must be an object")
    if name == "gather":
        raise NativeBoundaryViolation("external network collection is outside this worker's official item")
    if arguments.get("allowAnyPath") is True:
        raise NativeBoundaryViolation("allowAnyPath cannot expand the worker filesystem boundary")
    roots = tuple(path.resolve() for path in roots)

    def check_path(value: Any, pointer: str) -> None:
        if not isinstance(value, str) or not value:
            return  # Native schema validation owns malformed/empty arguments.
        raw = value.split("#", 1)[0]
        if raw == "~" or raw.startswith("~/"):
            candidate = home / raw[2:] if raw != "~" else home
        elif raw.startswith("~"):
            raise NativeBoundaryViolation(f"another user's home is outside worker scope: {pointer}")
        else:
            candidate = Path(raw)
            if not candidate.is_absolute():
                candidate = cwd / candidate
        resolved = candidate.resolve()
        if not any(resolved.is_relative_to(root) for root in roots):
            raise NativeBoundaryViolation(f"filesystem argument escapes this worker: {pointer}")

    def visit(value: Any, pointer: str = "$") -> None:
        if not isinstance(value, dict):
            return
        for key, entry in value.items():
            here = f"{pointer}.{key}"
            if key in {"repoPath", "repo_path", "cwd", "path", "root", "file"}:
                check_path(entry, here)
            elif key in {"files", "changed_files"} and isinstance(entry, list):
                for index, path in enumerate(entry):
                    check_path(path, f"{here}[{index}]")
            elif key in {"context", "params"}:
                visit(entry, here)
            elif key == "anchors" and isinstance(entry, list):
                for index, anchor in enumerate(entry):
                    visit(anchor, f"{here}[{index}]")

    visit(arguments)
