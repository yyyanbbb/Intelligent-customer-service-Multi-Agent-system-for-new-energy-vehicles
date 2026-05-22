#!/usr/bin/env python3
"""Unified project entrypoint.

This repository now exposes a single project with two capabilities:
1) Task-oriented EV agent (`task`)
2) Automotive customer-service multi-agent (`cs`)
3) EV NER extraction subsystem (`ner`)
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


def _run_cs(argv: list[str]) -> int:
    """Delegate to cs_agent CLI."""
    cmd = [sys.executable, "-m", "cs_agent.main", *argv]
    return subprocess.call(cmd, cwd=str(ROOT))


def _run_task(argv: list[str]) -> int:
    """Delegate to task_agent CLI."""
    cmd = [sys.executable, "-m", "task_agent.main", *argv]
    return subprocess.call(cmd, cwd=str(ROOT))


def _run_ner(argv: list[str]) -> int:
    """Delegate to ev_ner_agent CLI."""
    cmd = [sys.executable, "-m", "ev_ner_agent.main", *argv]
    return subprocess.call(cmd, cwd=str(ROOT))


def _print_help() -> None:
    print("Unified EV AI project launcher (task + cs + ner).")
    print("")
    print("Usage:")
    print("  python run.py [task|cs|ner] [args...]")
    print("")
    print("Examples:")
    print("  python run.py task --query \"我想买台电车\"")
    print("  python run.py cs --ui")
    print("  python run.py cs --eval")
    print("  python run.py ner --interactive")


def main() -> int:
    argv = sys.argv[1:]
    if not argv:
        return _run_task([])

    if argv[0] in {"-h", "--help"}:
        _print_help()
        return 0

    mode = "task"
    if argv[0] in {"task", "cs", "ner"}:
        mode = argv[0]
        argv = argv[1:]

    if mode == "task":
        return _run_task(argv)
    if mode == "cs":
        return _run_cs(argv)
    return _run_ner(argv)


if __name__ == "__main__":
    raise SystemExit(main())
