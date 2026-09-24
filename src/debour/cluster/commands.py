"""Safely print or execute commands assembled by experiment runners.

Purpose:
    Give experiment-specific scripts one shared execution contract. Commands are
    represented as argument lists, displayed with shell-safe quoting, and only
    launched when the caller explicitly requests execution. This keeps dry-run
    planning and real execution identical without using ``shell=True``.

How to use:
    ``run_command(["python3", "workflows/train_ensemble.py", ...],
    execute=args.execute, cwd=repository_root)``. With ``execute=False`` the
    command is printed only; with ``True`` it is run and failures are propagated.
"""

from __future__ import annotations

import shlex
import subprocess
from pathlib import Path
from typing import Sequence


def format_command(command: Sequence[str]) -> str:
    """Return a copyable shell representation of an argument vector."""

    return shlex.join([str(part) for part in command])


def run_command(
    command: Sequence[str],
    *,
    execute: bool,
    cwd: str | Path,
) -> None:
    """Print a command and optionally execute it without a shell."""

    normalized = [str(part) for part in command]
    print(f"$ {format_command(normalized)}", flush=True)
    if execute:
        subprocess.run(normalized, cwd=Path(cwd), check=True)
