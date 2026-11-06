"""MCP-owned packets and a bounded output gate, shared by memory evaluations.

The compiler is called inside session's memory packet, never as a sidecar here.
The scorer receives the returned answer string; receipt/log data stays outside
the scorer boundary. Synthetic integration tests exercise this adapter.
"""
from .admit import admit


def log_from_sessions(sessions, *, binary):
    statements = []
    for index, session in enumerate(sessions, 1):
        statements.append({"id": session["id"], "seq": index,
                           "text": session["text"], "tags": session.get("tags", []),
                           "source_date": session.get("source_date"),
                           "stability_q": session.get("stability_q")})
    return admit(statements, binary=binary)


def read_with_gate(tool, generate, *, question, symbols=(), today=None):
    packet = tool("session", {"action": "start", "memory_hook": {
        "prompt": question, "symbols": list(symbols), "today": today, "limit": 5}})
    receipt = packet["compile_receipt"]  # Required: skipping compilation fails.
    preflight = receipt.get("gate", "Allow")
    if isinstance(preflight, dict) and "Abstain" in preflight:
        return preflight["Abstain"]["answer"]
    options = packet["compile_options"]
    draft = generate(packet["reader_context"], question)
    for attempt in range(2):
        checked = tool("precedent", {"query": "compile_receipt",
            "log_offset": receipt["offset"], "compile_options": options,
            "draft": draft, "expected_receipt_hash": receipt["hash"]})
        verdict = checked["verdict"]
        if verdict == "Allow":
            return draft
        if isinstance(verdict, dict) and "Abstain" in verdict:
            return verdict["Abstain"]["answer"]
        if verdict == "SliceMismatch":
            raise RuntimeError("The packet changed or no longer passes verification")
        if attempt:
            raise RuntimeError("The retried answer still failed the exact-evidence gate")
        pointers = next(iter(verdict.values()))["controlling"]
        draft = generate(packet["reader_context"] + "\nRETRY ONCE: use these exact "
                         "controlling records; verify current/historical scope: "
                         + ", ".join(pointers), question)
    raise AssertionError("unreachable")
