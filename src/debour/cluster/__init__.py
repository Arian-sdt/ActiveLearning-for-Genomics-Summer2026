"""Slurm generation and safe experiment-command execution.

Purpose: separate resource requests and orchestration from scientific code.
How to use: import ``run_command`` or ``write_sbatch``.
"""

from .commands import format_command, run_command
from .slurm import SlurmResources, write_sbatch

__all__ = ["SlurmResources", "format_command", "run_command", "write_sbatch"]
