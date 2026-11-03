"""Metered model transport for original hooks and the harness agent loop.

This module implements no memory. Hook tools call the same live child's MCP
client, and every model request uses the shared campaign provider/ledger.
"""
from __future__ import annotations
import asyncio
import json
from pathlib import Path

from .provider import MODEL
from .runtime import append_jsonl,canonical


class ItemModelBudget(RuntimeError):
    pass


class HookCallback:
    def __init__(self,bound,engine_getter=None):
        self.bound=bound
        self.engine_getter=engine_getter
        self.engine=None
        self.usage=[]
        self.cost=0.0

    def bind_engine(self,engine):
        self.engine=engine

    def __call__(self,payload):
        pending=asyncio.run_coroutine_threadsafe(self.bound.hook_request(payload,self),self.bound.loop)
        try:
            return pending.result(timeout=180)
        except BaseException:
            pending.cancel()
            raise


class BudgetedProvider:
    def __init__(self,provider,meta,root,*,max_calls=6,max_output=2048,context=32768):
        self.provider,self.meta,self.root=provider,dict(meta),Path(root)
        self.loop=asyncio.get_running_loop()
        self.remaining_calls,self.remaining_output=max_calls,max_output
        self.limits={"max_calls":max_calls,"max_output":max_output,"context":context}
        self.context=context
        self.cost=0.0
        self.usage=[]
        self.lock=asyncio.Lock()

    async def chat(self,messages,*,tools=None,max_tokens=512,log=None,identity=None):
        from .run import tokens
        async with self.lock:
            if self.remaining_calls<=0 or self.remaining_output<=0:
                raise ItemModelBudget("equal item model-call/output budget exhausted (hooks included)")
            if tokens({"messages":messages,"tools":tools or []})>self.context:
                raise ItemModelBudget("equal item context budget exceeded (hooks included)")
            append_jsonl(self.root/"budget.jsonl",{"event":"budget_before","identity":self.meta,
                "limits":self.limits,"remaining_calls":self.remaining_calls,"remaining_output":self.remaining_output})
            result=await self.provider.chat(messages,tools=tools,
                        max_tokens=min(max_tokens,self.remaining_output),
                        log=log or self.root/"api.jsonl",identity={**self.meta,**(identity or {})})
            self.remaining_calls-=1
            self.remaining_output-=result["usage"]["completion_tokens"]
            self.cost+=result["cost"]
            self.usage.append(result["usage"])
            append_jsonl(self.root/"budget.jsonl",{"event":"budget_after","identity":self.meta,
                "limits":self.limits,"remaining_calls":self.remaining_calls,"remaining_output":self.remaining_output,
                "usage":result["usage"],"aggregate_cost":self.cost})
            return result

    def hook_callback(self,engine_getter=None):
        return HookCallback(self,engine_getter)

    async def hook_request(self,payload,callback):
        if payload.get("model")!=MODEL:
            raise RuntimeError("hook transport requested a different solver model")
        messages=list(payload["messages"])
        engine=callback.engine_getter() if callback.engine_getter else callback.engine
        allowed=payload.get("allowed_tools") or []
        actual={t["name"]:t for t in (engine.tools if engine else [])}
        schemas=[]
        mapping={}
        for requested in allowed:
            native=requested.removeprefix("mcp__vestige__")
            if native in actual:
                t=actual[native]
                mapping[requested]=native
                schemas.append({"type":"function","function":{"name":requested,
                    "description":t.get("description",""),"parameters":t["inputSchema"]}})
            else:
                append_jsonl(self.root/"hook-model.jsonl",{"event":"native_tool_unavailable",
                              "tool":requested,"hook":payload.get("hook")})
        max_tools=min(3,int(payload.get("tool_call_budget",0)))
        calls_used=0
        while True:
            result=await self.chat(messages,tools=schemas or None,
                    max_tokens=min(1024,int(payload.get("max_tokens",1024))),
                    log=self.root/"hook-api.jsonl",
                    identity={"phase":"hook","hook":payload.get("hook")})
            callback.cost+=result["cost"]
            callback.usage.append(result["usage"])
            message=result["message"]
            messages.append({k:v for k,v in message.items() if k in {"role","content","tool_calls"}})
            toolcalls=message.get("tool_calls") or []
            if not toolcalls:
                return {"model":result["model"],"choices":[{"message":message,
                    "finish_reason":result["finish_reason"]}],"usage":result["usage"]}
            for call in toolcalls:
                calls_used+=1
                if calls_used>max_tools:
                    raise ItemModelBudget("source hook's tool-call budget exhausted")
                requested=call["function"]["name"]
                if requested not in mapping or not engine:
                    raise RuntimeError("hook requested a tool absent from its native allowlist")
                args=json.loads(call["function"]["arguments"])
                # Hook-internal MCP requests are transport operations, not
                # nested harness lifecycle events. The native audit still logs
                # every request/result; no host-side answer is synthesized.
                def native_call():
                    with engine._call_lock:
                        return engine._call(mapping[requested],args)
                output=await asyncio.to_thread(native_call)
                messages.append({"role":"tool","tool_call_id":call["id"],"content":canonical(output)})
