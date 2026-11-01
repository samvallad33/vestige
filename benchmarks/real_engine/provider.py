"""Metered OpenAI calls; the SDK's implicit retry layer is deliberately absent."""
from __future__ import annotations

import json
from pathlib import Path
import uuid
import httpx
from .runtime import AsyncAPIPool, BudgetLedger, append_jsonl, canonical, utc_now

MODEL = "gpt-5.4-nano"
PRICING = {"input": 0.20, "cached_input": 0.02, "output": 1.25,
           "unit": "USD per million tokens", "checked": "2026-11-01",
           "source": "https://developers.openai.com/api/docs/models/gpt-5.4-nano"}


def load_key(path: Path) -> str:
    if path.stat().st_mode & 0o077:
        raise PermissionError("credential file must have owner-only permissions")
    for line in path.read_text().splitlines():
        if line.startswith("OPENAI_API_KEY="):
            key = line.partition("=")[2].strip()
            if key:
                return key
    raise RuntimeError("No OPENAI_API_KEY assignment in the selected local file")


def account(response: dict) -> tuple[float, dict]:
    usage = response.get("usage")
    if not usage or not all(k in usage for k in ("prompt_tokens", "completion_tokens")):
        raise RuntimeError("provider omitted usage; reservation retained, stop")
    cached = usage.get("prompt_tokens_details", {}).get("cached_tokens", 0)
    prompt, output = usage["prompt_tokens"], usage["completion_tokens"]
    if not 0 <= cached <= prompt or output < 0:
        raise RuntimeError("invalid usage; reservation retained")
    cost = ((prompt - cached) * PRICING["input"] + cached * PRICING["cached_input"]
            + output * PRICING["output"]) / 1e6
    return cost, usage


class NanoProvider:
    def __init__(self, runs: Path, ledger: BudgetLedger):
        self.runs, self.ledger = runs, ledger
        self.key = load_key(runs / ".env.openai")
        self.pool = AsyncAPIPool(ledger, max_inflight=20, max_retries=6)
        self.http = httpx.AsyncClient(timeout=httpx.Timeout(60.0),
                                     limits=httpx.Limits(max_connections=20, max_keepalive_connections=20))

    async def close(self):
        await self.http.aclose()

    async def chat(self, messages: list[dict], *, tools: list[dict] | None = None,
                   max_tokens: int = 512, log: Path, identity: dict) -> dict:
        payload = {"model": MODEL, "messages": messages, "max_completion_tokens": max_tokens,
                   "reasoning_effort": "none", "temperature": 0, "store": False}
        if tools:
            payload["tools"] = tools
            payload["parallel_tool_calls"] = False
        # UTF-8 byte count bounds BPE token count; 4096 covers message/tool
        # framing. Cache discounts are never assumed by reservations.
        upper_input = len(canonical(payload).encode("utf-8")) + 4096
        upper_cost = (upper_input * PRICING["input"] + max_tokens * PRICING["output"]) / 1e6
        call_id = f"{identity.get('benchmark','judge')}/{identity.get('arm','unassigned')}/{uuid.uuid4().hex}"
        append_jsonl(log, {"event": "request", "call_id": call_id, "timestamp": utc_now(),
                           "identity": identity, "payload": payload, "upper_cost": upper_cost})

        async def send():
            result = await self.http.post("https://api.openai.com/v1/chat/completions",
                                          headers={"Authorization": "Bearer " + self.key}, json=payload)
            result.raise_for_status()
            return result.json()

        response = await self.pool.request(call_id, upper_cost, send, account,
                     on_attempt=lambda row: append_jsonl(log, {"event": "attempt", **row}))
        cost, usage = account(response)
        append_jsonl(log, {"event": "response", "call_id": call_id, "timestamp": utc_now(),
                           "identity": identity, "response": response, "cost": cost, "usage": usage})
        return {"message": response["choices"][0]["message"], "cost": cost,
                "usage": usage, "model": response["model"], "call_id": call_id,
                "finish_reason": response["choices"][0]["finish_reason"]}
