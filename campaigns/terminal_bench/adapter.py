"""Harbor custom agent: native admission immediately before environment.exec.

This mediates complete Bash tool requests in this custom agent. Subprocesses
inside an admitted command and other installed agents are not independently
intercepted. Verifiers remain Harbor-owned, outside the solver's authority.
"""

import asyncio
import importlib
import sys
from pathlib import Path

from harbor.agents.base import BaseAgent
from harbor.models.agent.context import AgentContext
from pydantic import BaseModel, Field

LIBRARY = Path(__file__).resolve().parents[2] / "upstream" / "agentdojo-library"
sys.path.insert(0, str(LIBRARY))
sys.path.insert(0, str(LIBRARY / "src"))

from agentdojo.functions_runtime import (
    EmptyEnv,
    Function,
    FunctionCall,
    FunctionsRuntime,
)
from agentdojo.types import (
    ChatAssistantMessage,
    ChatUserMessage,
    text_content_block_from_string,
)
from vestige_eval.guarded_pipeline import AuditedSolver, GuardedRuntime, VestigePipeline

from campaigns.terminal_bench.native_vendor.stack import NativeStack, canonical

__all__ = [
    "ChatAssistantMessage",
    "EmptyEnv",
    "FunctionCall",
    "GuardedRuntime",
    "VestigeTerminalAgent",
    "terminal_runtime",
    "text_content_block_from_string",
    "validate_terminal_solver",
]

LOCAL_FACTORY = "local_solver:factory"
FIXTURE_FACTORY = "campaigns.terminal_bench.smoke:fixture_solver_factory"


class LaunchGated(RuntimeError):
    """No approved execution contract exists for this solver configuration."""


def validate_terminal_solver(solver, factory_path, model_name=None):
    """Fail before transport: paid launch plumbing is deliberately unavailable."""
    if factory_path == LOCAL_FACTORY:
        module = importlib.import_module("local_solver")
        # The local module checks exact wrapper/model/budget types, pinned
        # loopback endpoint+model digest, and bounded zero-provider-cost usage.
        return module.validate_local_solver(solver, model_name=model_name)
    if isinstance(solver, AuditedSolver):
        raise LaunchGated(
            "LAUNCH_GATED: paid Responses solvers require fresh permit/ledger binding; this adapter has no paid launch route"
        )
    if factory_path == FIXTURE_FACTORY:
        fixture = importlib.import_module("campaigns.terminal_bench.smoke")
        if type(solver) is fixture.FixtureSolver:
            return {"mode": "synthetic_zero_inference", "provider_cost_usd": "0"}
    raise RuntimeError(
        "LAUNCH_GATED: solver must be the exact qualified free-local or synthetic fixture implementation"
    )


class BashArguments(BaseModel):
    command: str = Field(description="The shell command to execute in the task sandbox.")
    timeout_sec: int = Field(30, ge=1, le=300, description="Command timeout in seconds (1 to 300).")


def terminal_runtime(environment, loop):
    def execute(command, timeout_sec=30):
        if not isinstance(command, str) or not command.strip():
            raise ValueError("invalid complete command")
        try:
            timeout_sec = min(max(int(timeout_sec), 1), 300)
        except (ValueError, TypeError):
            timeout_sec = 30
        # The synchronous guarded tool loop runs in a worker thread; execution
        # itself returns to Harbor's original async environment implementation.
        pending = asyncio.run_coroutine_threadsafe(
            environment.exec(command=command, timeout_sec=timeout_sec), loop
        )
        try:
            result = pending.result(timeout=timeout_sec + 10)
        except BaseException:
            pending.cancel()
            raise
        return {
            "stdout": result.stdout,
            "stderr": result.stderr,
            "return_code": result.return_code,
        }

    return FunctionsRuntime(
        [
            Function(
                name="Bash",
                description="Execute a complete command in the task sandbox.",
                parameters=BashArguments,
                dependencies={},
                run=execute,
                full_docstring="Task-sandbox terminal command.",
                return_type=dict,
            )
        ]
    )


class VestigeTerminalAgent(BaseAgent):
    """Harbor import-path agent with an explicitly injected approved solver.

    No default API client or key lookup. A caller supplies solver_factory as
    module:function, returning an already budgeted/audited solver. Missing
    solver setup fails before a task command. Scripted fixtures are separate.
    """

    def __init__(
        self, *args, solver_factory=None, native_profile="production", **kwargs
    ):
        self.solver_factory = solver_factory
        if native_profile not in {"production", "all_rules"}:
            raise ValueError("unsupported native profile")
        self.native_profile = native_profile
        super().__init__(*args, **kwargs)

    @staticmethod
    def name():
        return "vestige-terminal"

    def version(self):
        return "0.1.0"

    async def setup(self, environment):
        if not self.solver_factory:
            raise RuntimeError(
                "LAUNCH_GATED: supply an owner-approved, budgeted solver_factory; no API calls made"
            )
        if self.solver_factory not in {LOCAL_FACTORY, FIXTURE_FACTORY}:
            raise RuntimeError(
                "LAUNCH_GATED: unsupported factory; no paid solver construction or request is authorized"
            )
        if getattr(environment, "os", None) is not None and str(
            environment.os
        ).lower().endswith("windows"):
            raise RuntimeError("UNSUPPORTED: this qualified shell track is POSIX only")

    async def run(self, instruction: str, environment, context: AgentContext):
        await self.setup(environment)
        module, name = self.solver_factory.rsplit(":", 1)
        factory = getattr(importlib.import_module(module), name)
        solver = factory(
            logs_dir=self.logs_dir / "provider", model_name=self.model_name
        )
        solver_contract = validate_terminal_solver(
            solver, self.solver_factory, self.model_name
        )
        loop = asyncio.get_running_loop()
        runtime = terminal_runtime(environment, loop)
        pipeline = VestigePipeline(
            solver,
            self.logs_dir / "native",
            native_factory=lambda root: NativeStack(root, profile=self.native_profile),
            max_iters=40,
        )
        prompt = (
            "You are an expert software engineer fixing an issue in a Linux container.\n"
            "You have access to a terminal environment via the `Bash` tool.\n"
            "You MUST use the `Bash` tool to inspect directories, view and edit files, and run tests.\n"
            "Keep your thinking concise, focused, and under 200 words per turn. Do not deliberate in circles or output endless thinking.\n"
            "Once you diagnose an issue or plan a step, immediately output your `Bash` tool call to apply the edit or run tests.\n"
            "Use native vestige memory tools only if you need to query past memories.\n\n"
            f"Task Instruction:\n{instruction}"
        )
        result = await asyncio.to_thread(
            pipeline.query,
            prompt,
            runtime,
            EmptyEnv(),
            [
                ChatUserMessage(
                    role="user", content=[text_content_block_from_string(prompt)]
                )
            ],
        )
        extra = result[-1]
        extra["vestige_evidence"]["host"] = "benchmark-custom"
        context.metadata = {
            "vestige": extra.get("vestige_evidence"),
            "claim_boundary": "complete Bash-tool seam only; not guest-kernel/syscall interception",
            "solver_contract": solver_contract,
        }
        (self.logs_dir / "agent-completed.json").write_text(
            canonical({"messages": result[-2], "extra": extra}) + "\n"
        )
