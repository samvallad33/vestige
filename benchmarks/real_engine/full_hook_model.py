"""Full-hook native error forwarding; baseline provider behavior is inherited.

The hook loop retains the original budgets, tool mapping and messages. Only an
actual returned MCP isError envelope is forwarded as the corresponding tool
output. Process death and transport failures still abort.
"""
from __future__ import annotations

import asyncio
import json

from .engine import VestigeMCPToolError
from .hook_model import BudgetedProvider,ItemModelBudget
from .provider import MODEL
from .runtime import append_jsonl,canonical


class FullHookBudgetedProvider(BudgetedProvider):
    async def hook_request(self,payload,callback):
        if payload.get("model")!=MODEL:
            raise RuntimeError("hook transport requested a different solver model")
        messages=list(payload["messages"])
        engine=callback.engine_getter() if callback.engine_getter else callback.engine
        allowed=payload.get("allowed_tools") or []
        actual={t["name"]:t for t in (engine.tools if engine else [])}
        schemas=[];mapping={}
        for requested in allowed:
            native=requested.removeprefix("mcp__vestige__")
            if native in actual:
                t=actual[native];mapping[requested]=native
                schemas.append({"type":"function","function":{"name":requested,
                    "description":t.get("description",""),"parameters":t["inputSchema"]}})
            else:
                append_jsonl(self.root/"hook-model.jsonl",{"event":"native_tool_unavailable",
                              "tool":requested,"hook":payload.get("hook")})
        max_tools=min(3,int(payload.get("tool_call_budget",0)));calls_used=0
        while True:
            result=await self.chat(messages,tools=schemas or None,
                    max_tokens=min(1024,int(payload.get("max_tokens",1024))),
                    log=self.root/"hook-api.jsonl",identity={"phase":"hook","hook":payload.get("hook")})
            callback.cost+=result["cost"];callback.usage.append(result["usage"])
            message=result["message"]
            messages.append({k:v for k,v in message.items() if k in {"role","content","tool_calls"}})
            toolcalls=message.get("tool_calls") or []
            if not toolcalls:
                return {"model":result["model"],"choices":[{"message":message,
                    "finish_reason":result["finish_reason"]}],"usage":result["usage"]}
            for call in toolcalls:
                calls_used+=1
                if calls_used>max_tools:raise ItemModelBudget("source hook's tool-call budget exhausted")
                requested=call["function"]["name"]
                if requested not in mapping or not engine:
                    raise RuntimeError("hook requested a tool absent from its native allowlist")
                args=json.loads(call["function"]["arguments"])
                def native_call():
                    with engine._call_lock:
                        try:return engine._call(mapping[requested],args)
                        except VestigeMCPToolError as exc:
                            append_jsonl(self.root/"hook-model.jsonl",{
                                "event":"native_tool_error_forwarded","tool":mapping[requested],
                                "arguments":args,"raw_mcp_result":exc.result})
                            return exc.result
                output=await asyncio.to_thread(native_call)
                messages.append({"role":"tool","tool_call_id":call["id"],"content":canonical(output)})
