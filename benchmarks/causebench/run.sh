#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
PYTHONPATH=. python3 -m unittest discover -s tests -p 'test_*.py'
PYTHONPATH=. python3 evaluate.py
