"""Write portable Slurm job scripts without hard-coding unavailable nodes.

Purpose:
    Generate reproducible batch jobs while avoiding the historical failure mode
    of treating interactive pseudo-partitions as valid batch partitions. Partition
    is optional so Nibi can route a job; GPU GRES is user-configurable.

How to use:
    ``write_sbatch(path, command, SlurmResources(...))`` then run
    ``sbatch path``. Inspect jobs with ``squeue -u $USER`` and ``sacct -j JOBID``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class SlurmResources:
    job_name: str
    time: str = "01:00:00"
    cpus: int = 4
    memory: str = "24G"
    gres: str | None = None
    partition: str | None = None
    account: str | None = None


def write_sbatch(
    path: str | Path,
    command: str,
    resources: SlurmResources,
    *,
    project_dir: str = "~/projects/def-maxwl/aryansdt/DeBour",
    environment: str = "~/dream-rnn-env/bin/activate",
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    directives = [
        "#!/bin/bash",
        f"#SBATCH --job-name={resources.job_name}",
        f"#SBATCH --cpus-per-task={resources.cpus}",
        f"#SBATCH --mem={resources.memory}",
        f"#SBATCH --time={resources.time}",
        "#SBATCH --output=logs/%x_%j.out",
        "#SBATCH --error=logs/%x_%j.err",
    ]
    if resources.gres:
        directives.append(f"#SBATCH --gres={resources.gres}")
    if resources.partition:
        directives.append(f"#SBATCH --partition={resources.partition}")
    if resources.account:
        directives.append(f"#SBATCH --account={resources.account}")
    body = [
        *directives,
        "",
        "# Purpose: run one reproducible DeBour workflow in a non-interactive Slurm allocation.",
        "# How to run: sbatch this_file.sbatch; inspect with squeue/sacct and logs/*.out.",
        "set -euo pipefail",
        "module load python/3.11",
        f"source {environment}",
        f"cd {project_dir}",
        "mkdir -p logs",
        "",
        command,
        "",
    ]
    path.write_text("\n".join(body))
    return path

