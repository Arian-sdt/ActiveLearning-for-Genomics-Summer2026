# Project Notes

## Purpose

Summarize the project progression, methods tested, what worked, what failed, and the lessons that shaped this reusable package.

## How To Use

Use this as the narrative outline for a report or repository history. Cross-check final claims against `RESULTS.md` and regenerated artifacts.

## Phase 1: Baseline Model

- Reproduced a BHI-style CNN+BiLSTM for K562 activity regression.
- Used 4-channel one-hot DNA, Huber loss, AdamW, OneCycleLR cosine scheduling, 80 epochs, and best validation Pearson selection.
- Added exact split-ID persistence after discovering duplicate selected IDs could load 120,001 rows from a nominal 120,000-ID set.
- Trained five seeds on the same 120k base set to form an uncertainty ensemble.

**Worked:** robust training, checkpointing, and ensemble disagreement.

**Limitation:** validation splits differed by seed; ensemble members still shared the same overall data set, which was the essential requirement for uncertainty.

## Phase 2: Uncertainty Active Learning

- Predicted all unused Table S2 rows with five models.
- Ranked sample variance across predictions.
- Added 20k per round and retrained fresh 140k, 160k, and 180k models.
- Built random nested 20k/40k/60k baselines.
- Compared model sizes on shared unseen heldout sets.

**Worked:** performance rose across recorded uncertainty rounds.

**Lesson:** output all candidate variances, not only top IDs, so rankings and later analyses are reproducible.

## Phase 3: K-mer Distance And Facility Location

- Converted sequences to normalized 1/2/3-mer distributions.
- Implemented weighted Jensen-Shannon distance (`0.1/0.3/0.6`).
- Attempted exact on-demand facility location without precomputing an all-pairs matrix.

**Did not scale:** avoiding precomputation reduced storage but not the enormous number of distance calculations. Keeping the table in memory removed repeated I/O, not algorithmic cost.

## Phase 4: LSH Approximation

- Used random-hyperplane LSH over k-mer vectors.
- Analyzed 14, 18, and 22 bits; 22 bits yielded about 1,511 occupied buckets.
- Validated clusters by within-bucket JS distance against random same-size buckets.
- Used bucket centroids and sizes as a weighted facility ground set.
- Compared unconditional and base-conditional objectives.

**Worked:** LSH was fast and within-bucket distance was materially lower than random grouping.

**Caveat:** facility gains over centroids approximate full sequence-level gains.

## Phase 5: Hybrid Objectives

- Combined normalized ensemble variance with conditional/unconditional facility gain.
- Swept lambda values including `0.2,0.4,0.6,0.8` and small values down to `0.01`.
- Tested raw, initial-max, ground-size, and rank normalization strategies.
- Visualized selections over one reusable UMAP embedding.

**Issue:** facility marginal gains quickly collapsed while selected variance scores saturated near one. Small positive uncertainty weights could dominate. Dynamic rank normalization prevented ordinary lazy-greedy bounds and became expensive.

**Lesson:** inspect component distributions and selection trajectories, not only the formula or UMAP picture.

## Phase 6: Synthetic Redundancy Benchmark

- Selected diverse K562 prototypes using farthest-first weighted JS distance.
- Generated 50–100 related copies per parent with controlled mutation counts.
- Trained a BHI model on the full source table as an oracle and assigned synthetic activity labels.
- Repeated 10k-base, +2k acquisition experiments using random, uncertainty, conditional/unconditional family facility, and hybrid methods.
- Used parent families as the natural ground-set buckets.

**Worked:** redundancy and family identity became known rather than inferred.

**Important next control:** hold out entire parent families during evaluation; otherwise closely related copies can appear in both train and test.

## Phase 7: Evaluation And Operations

- Standardized shared test sets, paired bootstrap Pearson intervals, pairwise comparisons, learning curves, box/CI plots, and UMAP overlays.
- Used Slurm batch jobs and `tmux` for long runs.
- Diagnosed partition/GRES/node failures with `squeue`, `sacct`, `scontrol`, and `sinfo`.

**Operational lesson:** do not pin unavailable nodes, do not assume interactive partition labels accept `sbatch`, and do not interpret an empty `squeue` as success without checking `sacct` and error logs.

## Recommended Next Experiments

1. Repeat each acquisition method across multiple selection and training seeds.
2. Use family-heldout synthetic tests and report calibration metrics alongside Pearson r.
3. Compare centroid approximation against exact facility location on small pools where exact computation is feasible.
4. Measure overlap, coverage, label distribution, and family diversity of each selected set.
5. Pre-register normalization and lambda grids before evaluating test performance.

