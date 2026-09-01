"""Regression tests for CauseBench's published deterministic numbers."""

from __future__ import annotations

import socket
import unittest

from adapters.baselines import (
    EntityBlindVectorBaseline,
    FirstMemoryBaseline,
    LexicalOverlapBaseline,
)
from adapters.vestige import VestigeCausalBridge
from evaluate import score_adapter
from generator.generate import SEED, generate_tasks


class CauseBenchTests(unittest.TestCase):
    def test_generator_is_deterministic(self) -> None:
        self.assertEqual(SEED, 424242)
        first = generate_tasks()
        second = generate_tasks()
        self.assertEqual(first, second)
        self.assertEqual(len([task for task in first if task.split == "synthetic"]), 5)
        self.assertEqual(len([task for task in first if task.split == "real"]), 4)

    def test_expected_recall_at_1_numbers(self) -> None:
        tasks = generate_tasks()
        adapters = (
            VestigeCausalBridge(),
            FirstMemoryBaseline(),
            LexicalOverlapBaseline(),
            EntityBlindVectorBaseline(),
        )
        actual = {
            (score.adapter, score.split): score.percent
            for adapter in adapters
            for score in score_adapter(adapter, tasks)
        }
        self.assertEqual(actual[("vestige:causal-bridge", "synthetic")], 60)
        self.assertEqual(actual[("vestige:causal-bridge", "real")], 50)
        for adapter in (
            "baseline:first-memory",
            "baseline:lexical-overlap",
            "baseline:entity-blind-vector",
        ):
            self.assertEqual(actual[(adapter, "synthetic")], 0)
            self.assertEqual(actual[(adapter, "real")], 0)

    def test_no_network_is_required(self) -> None:
        def fail_socket(*args: object, **kwargs: object) -> socket.socket:
            raise AssertionError("CauseBench must not open network sockets")

        original_socket = socket.socket
        socket.socket = fail_socket  # type: ignore[assignment]
        try:
            tasks = generate_tasks()
            score_adapter(VestigeCausalBridge(), tasks)
            score_adapter(LexicalOverlapBaseline(), tasks)
        finally:
            socket.socket = original_socket  # type: ignore[assignment]


if __name__ == "__main__":
    unittest.main()
