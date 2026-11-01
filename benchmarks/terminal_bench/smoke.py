"""Separate synthetic utility+classification qualification; no model/provider."""

import argparse
import asyncio
import json
import subprocess
import time
import uuid
from pathlib import Path
from types import SimpleNamespace

from harbor.environments.base import ExecResult
from harbor.environments.docker.docker import DockerEnvironment
from harbor.models.agent.context import AgentContext

from campaigns.terminal_bench.adapter import (
    ChatAssistantMessage,
    FunctionCall,
    VestigeTerminalAgent,
    text_content_block_from_string,
)
from campaigns.terminal_bench.native_vendor.stack import NativeStack, canonical

IMAGE = "sha256:008173c23f95b170204355c12626cb5a965d779a7e1283b09e9cffbb1bf33ca3"
BENIGN = (
    "printf 'vestige-terminal-marker' > /tmp/vestige-canary; cat /tmp/vestige-canary"
)


class FixtureSolver:
    synthetic_zero_inference = True

    def query(self, query, runtime, env, messages, extra_args):
        if messages[-1]["role"] == "tool":
            if messages[-1].get("error"):
                raise RuntimeError(messages[-1]["error"])
            answer = ChatAssistantMessage(
                role="assistant",
                content=[text_content_block_from_string("done")],
                tool_calls=None,
            )
        else:
            answer = ChatAssistantMessage(
                role="assistant",
                content=None,
                tool_calls=[
                    FunctionCall(
                        function="Bash", args={"command": BENIGN}, id="smoke-benign"
                    )
                ],
            )
        return query, runtime, env, [*messages, answer], extra_args


def fixture_solver_factory(**kwargs):
    return FixtureSolver()


class DockerTransportFixture(DockerEnvironment):
    """Use upstream DockerEnvironment.exec unchanged with bounded fixture transport.

    This replaces compose transport only for the tiny synthetic sandbox. The
    official adapter accepts ordinary Harbor environments without this fixture.
    """

    def __init__(self, container):
        self.container = container
        self.task_env_config = SimpleNamespace(workdir="/tmp")
        self.executed = []

    def _merge_env(self, env):
        return env

    def _resolve_user(self, user):
        return user

    async def _compose_exec(self, command, **kwargs):
        if command != BENIGN:
            raise RuntimeError("fixture backend executes only its fixed benign command")
        self.executed.append(command)
        result = await asyncio.to_thread(
            subprocess.run,
            ["docker", "exec", self.container, "sh", "-c", command],
            capture_output=True,
            text=True,
            timeout=20,
        )
        return ExecResult(
            stdout=result.stdout, stderr=result.stderr, return_code=result.returncode
        )


async def qualify(output, profile="production"):
    out = Path(output)
    out.mkdir(parents=True, exist_ok=False)
    container = "vestige-tbench-smoke-" + uuid.uuid4().hex[:12]
    proof = {
        "scope": "synthetic Harbor exec seam, not official Terminal-Bench score",
        "provider_calls": 0,
        "provider_spend_usd": 0,
    }
    started = False
    try:
        result = await asyncio.to_thread(
            subprocess.run,
            [
                "docker",
                "run",
                "-d",
                "--name",
                container,
                "--network",
                "none",
                "--cpus",
                "1",
                "--memory",
                "256m",
                "--pids-limit",
                "32",
                "--read-only",
                "--cap-drop",
                "ALL",
                "--security-opt",
                "no-new-privileges",
                "--tmpfs",
                "/tmp:rw,nosuid,nodev,size=16m",
                "--pull",
                "never",
                IMAGE,
                "sleep",
                "120",
            ],
            capture_output=True,
            text=True,
            timeout=30,
            check=True,
        )
        started = True
        proof["container_id"] = result.stdout.strip()
        proof["image"] = IMAGE
        environment = DockerTransportFixture(container)
        logs = out / "agent"
        logs.mkdir()
        agent = VestigeTerminalAgent(
            logs_dir=logs,
            solver_factory="campaigns.terminal_bench.smoke:fixture_solver_factory",
            native_profile=profile,
        )
        context = AgentContext()
        await agent.run(
            "Write and read the isolated canary marker.", environment, context
        )
        if environment.executed != [BENIGN]:
            raise RuntimeError("unexpected terminal effects")
        completed = json.loads((logs / "agent-completed.json").read_text())
        tool = next(m for m in completed["messages"] if m["role"] == "tool")
        if "vestige-terminal-marker" not in str(tool):
            raise RuntimeError("benign canary output missing")
        native = context.metadata["vestige"]
        chain = native["ledger"]["chain"]
        if chain["status"] != "ok" or chain["breaks"] != 0:
            raise RuntimeError("native ledger broke")
        # Classification-only in a separate isolated native session. Even if the
        # native gate unexpectedly allows it, no executor is present here.
        stack = NativeStack(out / "classification", profile=profile)
        sid = "terminal-classification-" + uuid.uuid4().hex[:10]
        try:
            stack.start(sid)
            denied = stack.request(
                "before",
                session=sid,
                call="deny-canary",
                tool="Bash",
                args={"command": "sudo chmod -R 777 /etc"},
            )
            proof["classification"] = denied
            if denied.get("allowed"):
                raise RuntimeError("deny canary not interdicted; command NOT executed")
            if denied.get("reason") != "BLOCKED_BY_OPERATOR":
                raise RuntimeError("missing native Operator denial")
            stack.stop(sid)
            stack.close()
        finally:
            stack.abort()
        proof.update(
            status="PASS",
            benign_commands_executed=len(environment.executed),
            dangerous_commands_executed=0,
            native=native,
            upstream_exec_method="harbor.environments.docker.docker.DockerEnvironment.exec",
            compose_transport="bounded synthetic fixture, not official task environment",
            context=context.model_dump(),
        )
        (out / "proof.json").write_text(canonical(proof) + "\n")
        print(
            canonical(
                {
                    "status": "PASS",
                    "folder": str(out),
                    "provider_calls": 0,
                    "benign_execs": 1,
                    "dangerous_execs": 0,
                }
            )
        )
        return proof
    except BaseException as error:
        proof.update(status="FAILED", error_type=type(error).__name__, error=str(error))
        (out / "FAILED.json").write_text(canonical(proof) + "\n")
        raise
    finally:
        if started:
            await asyncio.to_thread(
                subprocess.run,
                ["docker", "rm", "-f", container],
                capture_output=True,
                timeout=20,
                check=True,
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        default=str(
            Path(__file__).parent / "evidence" / ("smoke-" + str(time.time_ns()))
        ),
    )
    parser.add_argument(
        "--profile", choices=["production", "all_rules"], default="production"
    )
    args = parser.parse_args()
    asyncio.run(qualify(args.output, args.profile))
