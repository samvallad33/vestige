"""Provider request serialization with explicit context-retention accounting."""

import copy
import json
from .packets import PacketCache, canonical, select_tool_definitions


class DeveloperSession:
    def __init__(
        self, catalog, mcp_call, *, instructions="", max_context_bytes=1_000_000
    ):
        if type(max_context_bytes) is not int or max_context_bytes < 1024:
            raise ValueError("max_context_bytes must be an integer >=1024")
        self.catalog = copy.deepcopy(catalog)
        self.mcp_call = mcp_call
        self.instructions = instructions
        self.max_context_bytes = max_context_bytes
        self.selected = {"memory_status"}
        select_tool_definitions(self.catalog, self.selected)
        self.cache = PacketCache()
        self.events = []
        self.epoch = 0

    def discover(self, names):
        selected = self.selected | set(names)
        definitions = select_tool_definitions(self.catalog, selected)
        self.selected = selected
        return definitions

    def add_user(self, text):
        self.events.append({"kind": "user", "text": str(text)})

    def add_assistant(self, text):
        self.events.append({"kind": "assistant", "text": str(text)})

    def reset_context(self, *, summary=None):
        """Call for compaction, checkpoint restore, or uncertain transcript retention."""
        self.events.clear()
        self.cache.clear()
        self.epoch += 1
        if summary is not None:
            self.add_user(summary)

    def retained_packets(self):
        return {
            event["result"]["packetId"]
            for event in self.events
            if event["kind"] == "tool"
            and event["result"].get("packetId")
            and event["result"].get("notModified") is False
        }

    def execute_tool(self, call_id, name, arguments):
        if name not in self.selected:
            raise ValueError("tool is not discovered: " + name)
        if any(e.get("call_id") == call_id for e in self.events):
            raise ValueError("duplicate tool call ID")
        args = copy.deepcopy(arguments)
        packet = name == "recall" and args.get("mode", "lookup") == "lookup"
        request = (
            self.cache.prepare(args, retained_packet_ids=self.retained_packets())
            if packet
            else args
        )
        result = self.mcp_call(name, request)
        # Accept standard MCP envelopes and already-decoded tool values.
        if isinstance(result, dict) and result.get("isError"):
            raise ValueError("MCP tool returned an error")
        if isinstance(result, dict) and "structuredContent" in result:
            result = result["structuredContent"]
        elif isinstance(result, dict) and isinstance(result.get("content"), list):
            texts = [
                block.get("text")
                for block in result["content"]
                if block.get("type") == "text"
            ]
            if len(texts) != 1:
                raise ValueError("expected one structured JSON tool result")
            result = json.loads(texts[0])
        if not isinstance(result, dict):
            raise ValueError("tool result must be an object")
        result = self.cache.accept(request, result) if packet else copy.deepcopy(result)
        self.events.append(
            {
                "kind": "tool",
                "call_id": call_id,
                "name": name,
                "arguments": args,
                "result": result,
            }
        )
        return copy.deepcopy(result)

    def request(self, provider, *, model, **options):
        """Build the exact SDK body. Caller passes it to its own SDK/capture seam.

        Refuse a context overflow rather than evict evidence behind the model's
        back. Reset/compact explicitly, then fetch packets again.
        """
        if provider not in ("openai_responses", "anthropic_messages"):
            raise ValueError("unsupported provider")
        reserved = {"input", "messages", "tools", "instructions", "system", "model"}
        if reserved & options.keys():
            raise ValueError("options may not replace session-owned context")
        definitions = select_tool_definitions(self.catalog, self.selected)
        events = copy.deepcopy(self.events)
        if provider == "openai_responses":
            messages = []
            for event in events:
                if event["kind"] in ("user", "assistant"):
                    messages.append({"role": event["kind"], "content": event["text"]})
                else:
                    messages.extend(
                        [
                            {
                                "type": "function_call",
                                "call_id": event["call_id"],
                                "name": event["name"],
                                "arguments": canonical(event["arguments"]),
                            },
                            {
                                "type": "function_call_output",
                                "call_id": event["call_id"],
                                "output": canonical(event["result"]),
                            },
                        ]
                    )
            tools = [
                {
                    "type": "function",
                    "name": t["name"],
                    "description": t.get("description", ""),
                    "parameters": t["inputSchema"],
                    "strict": False,
                }
                for t in definitions
            ]
            body = {
                "model": model,
                "instructions": self.instructions,
                "tools": tools,
                "input": messages,
                **options,
            }
        else:
            messages = []
            for event in events:
                if event["kind"] in ("user", "assistant"):
                    messages.append({"role": event["kind"], "content": event["text"]})
                else:
                    messages.extend(
                        [
                            {
                                "role": "assistant",
                                "content": [
                                    {
                                        "type": "tool_use",
                                        "id": event["call_id"],
                                        "name": event["name"],
                                        "input": event["arguments"],
                                    }
                                ],
                            },
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "type": "tool_result",
                                        "tool_use_id": event["call_id"],
                                        "content": canonical(event["result"]),
                                    }
                                ],
                            },
                        ]
                    )
            tools = [
                {
                    "name": t["name"],
                    "description": t.get("description", ""),
                    "input_schema": t["inputSchema"],
                }
                for t in definitions
            ]
            body = {
                "model": model,
                "system": self.instructions,
                "tools": tools,
                "messages": messages,
                **options,
            }
        if len(canonical(body).encode()) > self.max_context_bytes:
            raise ValueError(
                "context budget exceeded; compact explicitly and refresh evidence"
            )
        return body
