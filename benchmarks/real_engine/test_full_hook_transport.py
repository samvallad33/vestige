"""Essential full-hook transport tests; no API calls or benchmark scores."""
import asyncio
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from .engine import RealEngine,VestigeMCPError,VestigeMCPToolError
from .full_hook_model import FullHookBudgetedProvider
from .hook_model import HookCallback
from .hooks import HookViolation,TrackedHookCallback
from .runtime import read_jsonl

REPO=Path(__file__).resolve().parents[2]


class OwnedCallbackLifecycle(unittest.IsolatedAsyncioTestCase):
    async def test_pending_callback_is_cancelled_and_never_quiet_success(self):
        entered=asyncio.Event();cancelled=asyncio.Event();events=[];calls=[]
        bound=SimpleNamespace(loop=asyncio.get_running_loop(),cost=0.0,usage=[])
        async def request(payload,callback):
            calls.append(payload);entered.set()
            try:await asyncio.Event().wait()
            finally:cancelled.set()
        bound.hook_request=request
        tracked=TrackedHookCallback(HookCallback(bound),events.append)
        tracked.begin_event("fixture:UserPromptSubmit:0:0")
        invocation=asyncio.create_task(asyncio.to_thread(tracked,{"model":"gpt-5.4-nano"}))
        await asyncio.wait_for(entered.wait(),2)
        with self.assertRaisesRegex(HookViolation,"before its model transport"):
            await asyncio.wait_for(asyncio.to_thread(tracked.finish_event),2)
        await asyncio.gather(invocation,return_exceptions=True)
        self.assertTrue(cancelled.is_set())
        self.assertTrue(any(e["kind"]=="hook_transport_timeout" for e in events))
        await asyncio.to_thread(tracked.close)
        with self.assertRaises(HookViolation):
            await asyncio.to_thread(tracked,{"model":"gpt-5.4-nano"})
        self.assertEqual(len(calls),1)
        self.assertEqual(bound.cost,0)

    async def test_close_cancels_owned_request_and_preserves_bound_usage(self):
        entered=asyncio.Event();cancelled=asyncio.Event();events=[]
        bound=SimpleNamespace(loop=asyncio.get_running_loop(),cost=.003,usage=[{"total_tokens":10}])
        async def request(payload,callback):
            entered.set()
            try:await asyncio.Event().wait()
            finally:cancelled.set()
        bound.hook_request=request
        tracked=TrackedHookCallback(HookCallback(bound),events.append);tracked.begin_event("fixture:Stop:0:0")
        invocation=asyncio.create_task(asyncio.to_thread(tracked,{}))
        await asyncio.wait_for(entered.wait(),2)
        await asyncio.wait_for(asyncio.to_thread(tracked.close),2)
        await asyncio.gather(invocation,return_exceptions=True)
        self.assertTrue(cancelled.is_set())
        self.assertEqual(tracked.cost,.003);self.assertIs(tracked.usage,bound.usage)


@unittest.skipUnless((REPO/"runs/build-manifest.json").exists(),"actual release child required")
class ActualHookToolErrors(unittest.IsolatedAsyncioTestCase):
    async def exercise(self,dead=False):
        with tempfile.TemporaryDirectory(dir=REPO/"runs") as tmp:
            root=Path(tmp)
            with RealEngine(REPO,root/"native") as engine:
                bound=FullHookBudgetedProvider(None,{"benchmark":"unit-transport"},root)
                callback=HookCallback(bound);callback.bind_engine(engine);seen=[]
                async def chat(messages,**kwargs):
                    seen.append(json.loads(json.dumps(messages)))
                    message={"role":"assistant","content":"EMPTY"}
                    if len(seen)==1:
                        message={"role":"assistant","content":None,"tool_calls":[{"id":"unit-call","type":"function",
                            "function":{"name":"mcp__vestige__memory","arguments":json.dumps({"action":"invalid-action"})}}]}
                    return {"message":message,"cost":0.0,"usage":{"prompt_tokens":1,"completion_tokens":1},
                            "model":"gpt-5.4-nano","finish_reason":"stop"}
                bound.chat=chat
                payload={"model":"gpt-5.4-nano","messages":[{"role":"user","content":"Transport fixture"}],
                         "allowed_tools":["mcp__vestige__memory"],"tool_call_budget":3}
                if dead:
                    engine.client._proc.kill();engine.client._proc.wait(timeout=10)
                    with self.assertRaises(VestigeMCPError) as caught:await bound.hook_request(payload,callback)
                    self.assertNotIsInstance(caught.exception,VestigeMCPToolError)
                    self.assertEqual(len(seen),1)
                    self.assertFalse((root/"hook-model.jsonl").exists())
                else:
                    await bound.hook_request(payload,callback)
                    native=[e for e in read_jsonl(engine.audit) if e["kind"]=="tool_error"][-1]["raw_mcp_result"]
                    self.assertIs(native["isError"],True)
                    self.assertEqual(json.loads(seen[1][-1]["content"]),native)
                    self.assertEqual(read_jsonl(root/"hook-model.jsonl")[-1]["raw_mcp_result"],native)

    async def test_actual_native_iserror_reaches_hook_model_unchanged(self):await self.exercise()
    async def test_actual_engine_death_aborts_hook_model(self):await self.exercise(dead=True)


if __name__=="__main__":unittest.main()
