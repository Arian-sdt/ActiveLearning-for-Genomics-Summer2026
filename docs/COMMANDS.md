# Commands

## Purpose

Provide copyable local and Nibi commands for installation, training, selection, evaluation, and job inspection.

## How To Use

Run commands from `debour_pipeline/` unless the block explicitly changes directories. Replace placeholder paths with actual project-storage paths.

## Install And Test

```bash
module load python/3.11
source ~/dream-rnn-env/bin/activate
cd ~/projects/def-maxwl/aryansdt/DeBour/debour_pipeline
python3 -m pip install -e . --no-build-isolation
python3 -m pytest
```

## Train Five Models On One Shared 10k

```bash
python3 workflows/train_ensemble.py \
  --table ../data/DATA-Table_S2__MPRA_dataset.txt \
  --sample-rows 10000 \
  --selection-seed 2026 \
  --seeds 42 43 44 45 46 \
  --epochs 80 \
  --workers 4 \
  --output-dir outputs/base10k
```

## Score Remaining Data And Select Top 2k Variance

```bash
python3 workflows/select_sequences.py uncertainty \
  --table ../data/DATA-Table_S2__MPRA_dataset.txt \
  --model-runs outputs/base10k/seed42 outputs/base10k/seed43 \
    outputs/base10k/seed44 outputs/base10k/seed45 outputs/base10k/seed46 \
  --exclude-ids outputs/base10k/shared_ids.txt \
  --select-rows 2000 \
  --batch-size 512 \
  --workers 4 \
  --output-dir outputs/uncertainty2k
```

## Train Expanded 12k Ensemble

First create an ID file containing the base and selected IDs:

```bash
cat outputs/base10k/shared_ids.txt outputs/uncertainty2k/selected_ids.txt \
  | sort -u > outputs/uncertainty12k_ids.txt

python3 workflows/train_ensemble.py \
  --table ../data/DATA-Table_S2__MPRA_dataset.txt \
  --id-file outputs/uncertainty12k_ids.txt \
  --seeds 42 43 44 45 46 \
  --output-dir outputs/uncertainty12k
```

## Shared-Holdout Bootstrap Evaluation

```bash
python3 workflows/evaluate_models.py \
  --table ../data/DATA-Table_S2__MPRA_dataset.txt \
  --model-runs outputs/base10k/seed42 outputs/uncertainty12k/seed42 \
  --model-names base10k uncertainty12k \
  --test-rows 10000 \
  --test-seed 200 \
  --bootstrap-samples 1000 \
  --bootstrap-seed 42 \
  --output-dir outputs/evaluation_seed200
```

## Synthetic Families

```bash
python3 workflows/run_synthetic_benchmark.py \
  --table ../data/DATA-Table_S2__MPRA_dataset.txt \
  --prototype-rows 1000 \
  --mutation-levels 1 4 7 10 13 16 19 \
  --output-dir outputs/synthetic_mut1_19
```

## Select 2k With Known Synthetic Families

Use true parent families as the weighted facility ground set. This does not run
LSH. The base 10k is excluded in every mode.

Unconditional diversity baseline:

```bash
python3 workflows/select_family_facility.py \
  --synthetic-table outputs/synthetic_mut1_19/mutated_copies.tsv \
  --kmer-table outputs/synthetic_mut1_19/kmer_counts_normalized.tsv \
  --base-ids outputs/base10k/shared_ids.txt \
  --select-rows 2000 \
  --output-dir outputs/family_unconditional_2k
```

Conditional addition beyond base-set coverage:

```bash
python3 workflows/select_family_facility.py \
  --synthetic-table outputs/synthetic_mut1_19/mutated_copies.tsv \
  --kmer-table outputs/synthetic_mut1_19/kmer_counts_normalized.tsv \
  --base-ids outputs/base10k/shared_ids.txt \
  --conditional --select-rows 2000 \
  --output-dir outputs/family_conditional_2k
```

Hybrid conditional selection:

```bash
python3 workflows/select_family_facility.py \
  --synthetic-table outputs/synthetic_mut1_19/mutated_copies.tsv \
  --kmer-table outputs/synthetic_mut1_19/kmer_counts_normalized.tsv \
  --base-ids outputs/base10k/shared_ids.txt \
  --method hybrid --conditional \
  --variance-table outputs/synthetic_uncertainty/all_predictions.tsv \
  --lambda-uncertainty 0.5 --select-rows 2000 \
  --output-dir outputs/family_hybrid_conditional_2k
```

## Nibi Batch Template

Do not use `cpubase_interac` or `gpubase_interac` as batch partitions unless Nibi explicitly advertises them as submit-enabled. Omitting `--partition` lets the scheduler route the request.

```bash
mkdir -p logs
cat > train_ensemble.sbatch <<'EOF'
#!/bin/bash
#SBATCH --job-name=debour_ens
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
module load python/3.11
source ~/dream-rnn-env/bin/activate
cd ~/projects/def-maxwl/aryansdt/DeBour/debour_pipeline

python3 workflows/train_ensemble.py \
  --table ../data/DATA-Table_S2__MPRA_dataset.txt \
  --sample-rows 10000 --seeds 42 43 44 45 46 \
  --output-dir outputs/base10k
EOF
sbatch train_ensemble.sbatch
```

## Resource And Job Inspection

```bash
sinfo -o "%P %G %D %m %C"
sinfo -N -h -o "%N %T %G" | grep -E "idle|mix" | grep gpu | sort -u
squeue -u "$USER"
squeue --start -j JOB_ID
sacct -j JOB_ID --format=JobID,JobName,State,ExitCode,Elapsed,NodeList
scontrol show job JOB_ID
tail -f logs/JOB_NAME_JOB_ID.out
```

An empty `squeue -u $USER` means no pending or running jobs. Completed/failed jobs remain visible through `sacct`.
