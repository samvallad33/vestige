"""Strict offline artifact and pinned-source helpers shared by memory campaigns."""

import ast
import hashlib
import importlib
import json
import os
import subprocess
from pathlib import Path


def canonical(value):
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _object(pairs):
    out = {}
    for key, value in pairs:
        if key in out:
            raise ValueError("duplicate JSON field: " + key)
        out[key] = value
    return out


def read_json(path):
    return json.loads(
        Path(path).read_text(),
        object_pairs_hook=_object,
        parse_constant=lambda value: (_ for _ in ()).throw(
            ValueError("nonfinite JSON: " + value)
        ),
    )


def read_rows(path):
    path = Path(path)
    if path.suffix == ".parquet":
        # Imported only when data is Parquet; this package does not install it.
        parquet = importlib.import_module("pyarrow.parquet")
        return parquet.read_table(path).to_pylist()
    value = read_json(path)
    if not isinstance(value, list):
        raise TypeError("dataset must be a JSON array or Parquet table")
    return value


def source_identity(upstream, expected, paths, url):
    upstream = Path(upstream).resolve()
    revision = subprocess.check_output(
        ["git", "-C", str(upstream), "rev-parse", "HEAD"], text=True
    ).strip()
    if revision != expected:
        raise ValueError("upstream revision drift: " + revision)
    if subprocess.check_output(
        ["git", "-C", str(upstream), "status", "--porcelain", "--untracked-files=no"],
        text=True,
    ):
        raise ValueError("upstream tracked source is modified")
    return {
        "url": url,
        "revision": revision,
        "license": "MIT",
        "files": {name: sha(upstream / name) for name in paths},
    }


def extract_functions(path, names, namespace=None):
    """Load specified upstream pure function definitions, never module top-level.

    This avoids importing provider clients, dotenv, or heavyweight model modules.
    Source identity is pinned and recorded by preparation.
    """
    tree = ast.parse(Path(path).read_text(), filename=str(path))
    selected = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    if {node.name for node in selected} != set(names):
        raise ValueError("required upstream pure functions missing")
    scope = {} if namespace is None else dict(namespace)
    # Only pinned, unmodified upstream function definitions are used by callers;
    # executing their module would initialize provider/env dependencies instead.
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), "exec"), scope)  # noqa: S102
    return {name: scope[name] for name in names}


def constants(path, names):
    tree = ast.parse(Path(path).read_text())
    result, known = {}, {}

    class Resolve(ast.NodeTransformer):
        def visit_Name(self, node):
            if node.id in known:
                return ast.copy_location(
                    ast.parse(repr(known[node.id]), mode="eval").body, node
                )
            return node

    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    try:
                        value = ast.literal_eval(Resolve().visit(node.value))
                    except (ValueError, TypeError):
                        continue
                    known[target.id] = value
                    if target.id in names:
                        result[target.id] = value
    if set(result) != set(names):
        raise ValueError("upstream constant missing")
    return result


def write_packet(output, history, questions, oracle, manifest):
    out = Path(output)
    out.mkdir(parents=True, exist_ok=False, mode=0o700)
    if not questions:
        raise ValueError("empty prepared question set")
    if len({q["question_id"] for q in questions}) != len(questions):
        raise ValueError("duplicate question identities")
    for filename, records in (
        ("history.jsonl", history),
        ("questions.jsonl", questions),
        ("oracle.jsonl", oracle),
    ):
        with (out / filename).open("x", encoding="utf-8") as stream:
            os.chmod(out / filename, 0o600)
            for record in records:
                stream.write(canonical(record) + "\n")
    manifest.update(
        schema=1,
        status="PREPARED_ZERO_INFERENCE",
        provider_requests=0,
        cost_usd=0,
        paid_run_authorized=False,
        histories=len(history),
        questions=len(questions),
        outputs={
            name: sha(out / name)
            for name in ("history.jsonl", "questions.jsonl", "oracle.jsonl")
        },
        isolation={
            "solver_files": ["history.jsonl", "questions.jsonl"],
            "evaluator_only": ["oracle.jsonl"],
            "oracle_tags_forbidden": True,
            "handle_discovery": "runtime write receipts and neutral ordered session/chunk handles",
        },
    )
    manifest["sha256"] = hashlib.sha256(canonical(manifest).encode()).hexdigest()
    (out / "manifest.json").write_text(canonical(manifest) + "\n")
    return manifest


def jsonl(path):
    return [
        json.loads(line, object_pairs_hook=_object)
        for line in Path(path).read_text().splitlines()
        if line.strip()
    ]


def check_predictions(predictions):
    out = {}
    for item in predictions:
        qid, hypothesis = item.get("question_id"), item.get("hypothesis")
        if (
            not isinstance(qid, str)
            or not qid
            or not isinstance(hypothesis, str)
            or qid in out
        ):
            raise ValueError("invalid or duplicate hypothesis identity")
        out[qid] = hypothesis
    return out
