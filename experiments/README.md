# Experiment Catalogue

## Purpose

This directory preserves the concrete studies performed during the DeBour
project without mixing their paths and parameter choices into reusable source
code. Every experiment folder contains:

- `run.py`: a thin stage runner that calls shared workflows;
- `experiment.yaml`: the historical design and defaults;
- `README.md`: purpose, stage order, outputs, and lessons learned.

The runners print commands by default. Add `--execute` before the stage name to
actually run them. This makes each experiment inspectable and prevents an old
study definition from launching expensive work accidentally.

## Layout

| Folder | Study |
|---|---|
| `real_k562_active_learning/` | 120k base ensemble, repeated 20k uncertainty additions, random controls, and shared-holdout learning curves. |
| `lsh_centroid_selection/` | Random-hyperplane LSH and weighted-centroid conditional/unconditional facility selection on real data. |
| `synthetic_family_active_learning/` | Known parent mutation families, 10k base ensemble, 2k acquisitions, and 12k method comparisons. |
| `hybrid_lambda_sweep/` | Conditional/unconditional uncertainty-facility mixtures across lambda values and both LSH/family ground sets. |

## Architecture Rule

Experiment files may choose paths, seeds, budgets, and comparison groups. They
must not reimplement k-mers, Jensen-Shannon distance, neural networks, greedy
selection, bootstrapping, or plotting. Reusable behavior belongs in
`../src/debour/`; reusable command-level composition belongs in `../workflows/`.

## How To Use

From the repository root:

```bash
python3 -m pip install -e . --no-build-isolation
python3 experiments/synthetic_family_active_learning/run.py --help
```

Review a command without running it:

```bash
python3 experiments/real_k562_active_learning/run.py train-base \
  --table ../data/DATA-Table_S2__MPRA_dataset.txt
```

Launch it only after review:

```bash
python3 experiments/real_k562_active_learning/run.py --execute train-base \
  --table ../data/DATA-Table_S2__MPRA_dataset.txt
```
