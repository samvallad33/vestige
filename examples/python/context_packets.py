"""Client-side reuse for Vestige lookup packets, with explicit context retention.

This is an opt-in integration helper, not a patch to Codex or a provider cache.
The host must accurately report which packet IDs remain in model context.
"""
import copy
import json


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


class PacketCache:
    def __init__(self):
        self._packets = {}

    @staticmethod
    def key(arguments):
        args = dict(arguments)
        args.pop("known_packet_id", None)
        args.pop("knownPacketId", None)
        args.pop("contextPacket", None)
        args["context_packet"] = True
        return canonical(args)

    def prepare(self, arguments, *, retained_packet_ids=()):
        if arguments.get("mode", "lookup") != "lookup":
            raise ValueError("packets require lookup mode")
        args = dict(arguments)
        args.pop("known_packet_id", None)
        args.pop("knownPacketId", None)
        args.pop("contextPacket", None)
        args["context_packet"] = True
        entry = self._packets.get(self.key(args))
        if entry is not None and entry["packetId"] in retained_packet_ids:
            args["known_packet_id"] = entry["packetId"]
        return args

    def accept(self, request, response):
        """Return a small model-facing update. Never silently reuse lost context."""
        key = self.key(request)
        if response.get("notModified") is True:
            entry = self._packets.get(key)
            if (entry is None or request.get("known_packet_id") != response.get("packetId")
                    or entry["packetId"] != response.get("packetId")):
                raise ValueError("unacknowledged unchanged packet; retry without known_packet_id")
            return {"packetId": entry["packetId"], "notModified": True}
        packet_id = response.get("packetId")
        complete = (isinstance(packet_id, str) and len(packet_id) == 64
                    and all(char in "0123456789abcdef" for char in packet_id)
                    and response.get("evidenceIncomplete") is False)
        cards = response.get("results", [])
        if not isinstance(cards, list):
            raise ValueError("packet results must be an array")
        update = {"packetId": packet_id if complete else None, "notModified": False,
                  "evidenceIncomplete": not complete, "results": copy.deepcopy(cards)}
        # Dissent metadata belongs with the evidence, not in an optional log.
        if "contradictionProtected" in response:
            update["contradictionProtected"] = copy.deepcopy(response["contradictionProtected"])
        if complete:
            self._packets[key] = copy.deepcopy(update)
        else:
            self._packets.pop(key, None)
        return update

    def clear(self):
        """Call after compaction, a new conversation, or uncertain retention."""
        self._packets.clear()


def select_tool_definitions(catalog, names):
    """Choose exact native schemas for a client that supports dynamic catalogs.

    Tool selection is caller policy. This does not prove the model can discover
    omitted tools; retain memory_status and implement discovery in the host.
    """
    indexed = {tool["name"]: tool for tool in catalog}
    if len(indexed) != len(catalog):
        raise ValueError("duplicate tool definitions")
    requested = set(names)
    missing = requested - indexed.keys()
    if missing:
        raise ValueError("unknown tools: " + ", ".join(sorted(missing)))
    return [copy.deepcopy(indexed[name]) for name in sorted(requested)]
