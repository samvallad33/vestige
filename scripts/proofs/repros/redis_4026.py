#!/usr/bin/env python3
"""redis-py #4026: a refused TCP connect stays fast on 7.1.0 and blocks on retries later.

Exit 0 when the attempt returns in under a second. Exit 1 when retries push it
past that. Exit 125 when this tree cannot be imported or has no Redis client.
The port is bound and then closed so the kernel refuses it. Nothing here names
a commit.
"""

import os
import socket
import sys
import time

sys.path.insert(0, os.getcwd())

try:
    import redis
except Exception as exc:
    print(f"cannot import redis: {exc}", file=sys.stderr)
    sys.exit(125)

sock = socket.socket()
sock.bind(("127.0.0.1", 0))
port = sock.getsockname()[1]
sock.close()

# One attempt is not decisive: the default backoff is jittered, so a single
# refused connect on the regressing commit can return in under a second or
# in several. Five attempts make the two ends land on opposite sides of the
# cut. A good tree refuses the port in about a millisecond each time.
slow = 0
total = 0.0
for _ in range(5):
    started = time.perf_counter()
    try:
        redis.Redis(
            host="127.0.0.1",
            port=port,
            socket_connect_timeout=0.2,
            socket_timeout=0.2,
        ).ping()
    except Exception:
        pass
    elapsed = time.perf_counter() - started
    total += elapsed
    if elapsed >= 0.25:
        slow += 1
print(f"attempts=5 slow={slow} total={total:.3f}s")
sys.exit(0 if slow == 0 and total < 0.5 else 1)
