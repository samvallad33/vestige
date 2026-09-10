import os
from pathlib import Path
import sys
import time
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vestige_runtime import StdioMcp


@unittest.skipUnless(os.name == "posix", "POSIX transport")
class StdioTests(unittest.TestCase):
    def test_initialization_timeout_closes_owned_process(self):
        transport = StdioMcp(
            [sys.executable, "-I", "-c", "import time; time.sleep(30)"], timeout=0.1
        )
        with self.assertRaises(TimeoutError):
            with transport:
                pass
        self.assertIsNone(transport.process)

    def test_large_request_to_nonreading_server_obeys_deadline(self):
        server = (
            "import sys,json,time\n"
            "r=json.loads(sys.stdin.readline())\n"
            'print(json.dumps({"jsonrpc":"2.0","id":r["id"],"result":{}}),flush=True)\n'
            "time.sleep(30)\n"
        )
        with StdioMcp([sys.executable, "-I", "-c", server], timeout=0.2) as transport:
            started = time.monotonic()
            with self.assertRaises(TimeoutError):
                transport.call("fixture", {"payload": "x" * 1_000_000})
            self.assertLess(time.monotonic() - started, 2)

    def test_oversized_response_is_rejected_and_child_closed(self):
        server = 'import sys; sys.stdin.readline(); print("x"*2000,flush=True)'
        transport = StdioMcp(
            [sys.executable, "-I", "-c", server], max_response_bytes=1024
        )
        with self.assertRaises(ValueError):
            with transport:
                pass
        self.assertIsNone(transport.process)
