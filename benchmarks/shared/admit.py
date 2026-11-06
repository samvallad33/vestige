"""One admission implementation: Rust h2sk::admit::admit_statement.

This adapter takes source statements only. It does not accept evaluation labels,
relevance ids, benchmark names or question types, and never implements a second
suite-specific extractor.
"""
import json
import subprocess


def admit(statements, *, binary, prefix=None):
    allowed = {"id", "seq", "text", "tags", "source_date", "stability_q"}
    clean = []
    for statement in statements:
        if set(statement) - allowed:
            raise ValueError("Admission accepts source statement fields only")
        clean.append({"tags": [], "source_date": None, "stability_q": None, **statement})
    request = {"operation": "admit", "statements": clean}
    if prefix is not None:
        request["prefix"] = prefix
    result = subprocess.run([str(binary)], input=json.dumps(request), text=True,
                            capture_output=True, timeout=30, check=True)
    return json.loads(result.stdout)
