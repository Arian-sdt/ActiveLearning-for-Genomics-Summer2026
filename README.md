# DeBour Pipeline

## Purpose

This folder is a self-contained, reusable version of the DeBour DNA active-learning workflow. It covers sequence-table validation, normalized k-mer features, weighted Jensen-Shannon distance, the BHI-style CNN+BiLSTM model, ensemble uncertainty, facility-location and hybrid selection, bootstrap evaluation, visualization, synthetic redundancy benchmarks, and Slurm job generation. Reusable implementation lives in `src/debour/`; concrete historical studies live in `experiments/`.


## Project Roadmap

The pipeline is organized around the order in which an experiment is normally
performed. The numbered stages below point to the main files involved at each
step.

### Stage 1: Prepare And Validate Data

1. Load a Table S2-compatible tab-separated table.
2. Validate its identifier, sequence, and target columns.
3. Remove invalid rows and duplicate IDs.
4. Normalize every sequence to the model's expected length.
5. Create explicit sampled, training, validation, exclusion, and heldout ID sets.

Main files: `data/schema.py`, `data/io.py`, `data/sequences.py`, and
`data/splits.py`.

### Stage 2: Build Sequence Features

1. Count overlapping 1-mers, 2-mers, and 3-mers.
2. Normalize each k-mer block independently into a probability distribution.
3. Compare distributions with weighted Jensen-Shannon distance.
4. Optionally hash the feature vectors into LSH buckets and use weighted bucket
   centroids as a compressed facility-location ground set.

Main files: `features/kmers.py`, `features/jensen_shannon.py`, and
`selection/lsh.py`.

### Stage 3: Train A Base Ensemble

1. Select one shared pool of training IDs.
2. Train the BHI-style CNN+BiLSTM from scratch for several random seeds.
3. Save each model's exact splits, configuration, history, last checkpoint, and
   best-validation-Pearson checkpoint.

Main files: `models/bhi_cnn_bilstm.py`, `training/config.py`,
`training/trainer.py`, `training/ensemble.py`, and
`workflows/train_ensemble.py`.

### Stage 4: Score The Candidate Pool

1. Exclude every sequence already present in the training set.
2. Predict every remaining sequence with all ensemble members.
3. Store each model's prediction, the ensemble mean, sample variance, and
   standard deviation.

Main files: `models/checkpoints.py`,
`inference/ensemble_predictions.py`, and `workflows/select_sequences.py`.

### Stage 5: Select New Sequences

The repository supports four related acquisition strategies:

- **Random:** uniform baseline sampling.
- **Uncertainty:** highest ensemble prediction variance.
- **Facility location:** examples that best cover the sequence feature space.
- **Hybrid:** a convex combination of normalized uncertainty and facility gain.

Facility selection can be unconditional or conditional on coverage already
provided by the base training set. Conditional selection is usually the more
appropriate acquisition objective when adding examples to an existing model.

Main files: `selection/random.py`, `selection/uncertainty.py`,
`selection/facility_location.py`, `selection/families.py`, `selection/hybrid.py`,
`workflows/select_sequences.py`, `workflows/select_facility.py`, and
`workflows/select_family_facility.py`.

### Stage 6: Retrain From Scratch

1. Union the original IDs with the newly selected IDs.
2. Train fresh models rather than fine-tuning the old checkpoints.
3. Repeat the selection/retraining cycle for additional active-learning rounds.

Main file: `workflows/run_active_learning_round.py`.

### Stage 7: Evaluate Fairly

1. Exclude the union of all compared models' training IDs.
2. Draw one deterministic shared heldout set.
3. Generate predictions for every model on exactly those rows.
4. Calculate Pearson r, MSE, RMSE, and MAE.
5. Use paired bootstrap resampling for confidence intervals and pairwise tests.

Main files: `evaluation/holdout.py`, `evaluation/metrics.py`,
`evaluation/bootstrap.py`, and `workflows/evaluate_models.py`.

### Stage 8: Analyze And Visualize Results

Generate learning curves, bootstrap confidence-interval plots, facility-coverage
curves, and reusable UMAP overlays showing where each method samples sequence
space.

Main files: `visualization/performance.py`, `visualization/coverage.py`, and
`visualization/umap.py`.

### Stage 9: Run Controlled Synthetic Experiments

1. Choose diverse prototypes using farthest-first weighted JS distance.
2. Generate known redundant families by mutating each prototype.
3. Label synthetic sequences with a trained oracle model.
4. Compare random, uncertainty, family-facility, conditional facility, and
   hybrid selection under known family structure.

Main files: `synthetic/prototypes.py`, `synthetic/mutations.py`,
`synthetic/oracle_labels.py`, `selection/families.py`,
`workflows/run_synthetic_benchmark.py`, and
`workflows/select_family_facility.py`.

### Stage 10: Run On Nibi

Use generated or hand-written Slurm batch jobs, keeping resource requests
separate from scientific code. Avoid hard-coding node names and inspect completed
or failed jobs with `sacct`, not only `squeue`.

Main file: `cluster/slurm.py`. See `docs/COMMANDS.md` for Nibi examples.

### Stage 11: Reproduce A Historical Experiment

Choose a named folder under `experiments/`. Its `run.py` defines the concrete
stages, paths, budgets, and comparisons; its `experiment.yaml` records the study
design; and its README records what was learned. Experiment runners print their
commands by default and require `--execute` before they launch work.

Main folders: `experiments/real_k562_active_learning/`,
`experiments/lsh_centroid_selection/`,
`experiments/synthetic_family_active_learning/`, and
`experiments/hybrid_lambda_sweep/`.

## Repository Map

### Root Files

| File | Purpose |
|---|---|
| `README.md` | Main entry point, workflow roadmap, installation guide, and quick-start commands. |
| `pyproject.toml` | Defines the installable Python package, dependencies, optional extras, and command-line entry points. |
| `.gitignore` | Prevents checkpoints, generated outputs, logs, caches, and local environments from being committed. |

### Source And Experiment Boundaries

| Location | Purpose |
|---|---|
| `src/debour/` | Reusable data, feature, model, training, inference, selection, evaluation, visualization, synthetic, and cluster code. |
| `workflows/` | Reusable command-level compositions of source modules. |
| `experiments/` | Study-specific stage runners, manifests, historical defaults, and interpretations. |
| `src/README.md` | Rules for deciding whether new code belongs in source, workflows, or experiments. |
| `experiments/README.md` | Catalogue and usage guide for historical experiments. |

### `workflows/`: Commands Users Run

These scripts are deliberately thin. They parse command-line arguments and
compose reusable functions from `src/debour/`.

| File | Purpose |
|---|---|
| `workflows/train_ensemble.py` | Samples or loads one exact ID set and trains independently seeded CNN+BiLSTM models on it. |
| `workflows/select_sequences.py` | Runs random selection or full-pool ensemble-variance scoring and uncertainty selection. |
| `workflows/select_facility.py` | Builds weighted LSH centroids and runs conditional/unconditional facility or hybrid selection. |
| `workflows/select_family_facility.py` | Replaces LSH with known synthetic `parent_id` families and runs unconditional, conditional, or hybrid selection over weighted family centroids. |
| `workflows/run_active_learning_round.py` | Performs one complete uncertainty round: score, select, combine IDs, and retrain a fresh ensemble. |
| `workflows/evaluate_models.py` | Builds a shared unseen holdout, predicts with all models, bootstraps Pearson r, and saves results and a graph. |
| `workflows/run_synthetic_benchmark.py` | Selects diverse prototypes and creates controlled mutation families for synthetic experiments. |

### `src/debour/data/`: Data Contracts And ID Management

| File | Purpose |
|---|---|
| `data/__init__.py` | Exposes the most commonly used data APIs. |
| `data/schema.py` | Defines configurable names for ID, sequence, and target columns; defaults to `IDs`, `sequence`, and `K562_log2FC`. |
| `data/io.py` | Reads ID files and large tables, cleans chunks, validates rows, filters IDs, deduplicates, and writes ordered ID files. |
| `data/sequences.py` | Validates DNA, pads/crops sequences, computes GC content, and one-hot encodes channels in `A,G,C,T` order. |
| `data/splits.py` | Creates deterministic samples and train/validation splits and checks sets for leakage. |

### `src/debour/features/`: Sequence Representations

| File | Purpose |
|---|---|
| `features/__init__.py` | Exposes the standard k-mer and JS-distance functions. |
| `features/kmers.py` | Generates 84 normalized 1-mer, 2-mer, and 3-mer features from DNA sequences. |
| `features/jensen_shannon.py` | Computes bounded JS distance per k and combines distances with the project weights `0.1,0.3,0.6`. |

### `src/debour/models/`: Neural Network Definition

| File | Purpose |
|---|---|
| `models/__init__.py` | Exposes the model and checkpoint loader. |
| `models/bhi_cnn_bilstm.py` | Defines the 6,807,489-parameter parallel-CNN, bidirectional-LSTM regression architecture. |
| `models/checkpoints.py` | Finds and loads historical raw state dictionaries or richer training checkpoints. |

### `src/debour/training/`: Reproducible Model Training

| File | Purpose |
|---|---|
| `training/__init__.py` | Exposes the canonical configuration and training function. |
| `training/config.py` | Stores serializable defaults for epochs, batches, optimizer, scheduler, loss, split, seed, and device. |
| `training/trainer.py` | Implements datasets, training, validation, checkpointing, split persistence, and best-Pearson model selection. |
| `training/ensemble.py` | Repeats canonical training for several seeds on one exact shared data frame. |

### `src/debour/inference/`: Ensemble Predictions

| File | Purpose |
|---|---|
| `inference/__init__.py` | Exposes ensemble prediction. |
| `inference/ensemble_predictions.py` | Loads checkpoints, performs batched inference, and calculates per-row ensemble mean, variance, and standard deviation. |

### `src/debour/selection/`: Acquisition Algorithms

| File | Purpose |
|---|---|
| `selection/__init__.py` | Exposes family-centroid, facility, and hybrid selectors. |
| `selection/base.py` | Validates and saves a common ordered selection table, selected IDs, and configuration. |
| `selection/cli.py` | Provides the installed `debour-select` command for random or pre-scored uncertainty selection. |
| `selection/random.py` | Samples a reproducible random baseline without replacement. |
| `selection/uncertainty.py` | Sorts candidates by ensemble sample variance and records selection rank. |
| `selection/lsh.py` | Assigns random-hyperplane hash buckets and calculates their weighted feature centroids. |
| `selection/families.py` | Groups remaining synthetic copies by known parent, then returns stable family labels, mean k-mer centroids, and remaining-family-size weights. |
| `selection/facility_location.py` | Implements exact lazy-greedy facility selection over a full or centroid-compressed ground set. |
| `selection/hybrid.py` | Implements greedy selection combining min-max-normalized variance with ground-weight-normalized facility gain. |

### `src/debour/evaluation/`: Shared-Holdout Statistics

| File | Purpose |
|---|---|
| `evaluation/__init__.py` | Exposes common metrics and bootstrap analysis. |
| `evaluation/metrics.py` | Calculates Pearson r, MSE, RMSE, and MAE with degenerate-input checks. |
| `evaluation/holdout.py` | Discovers model training IDs and constructs a deterministic holdout unseen by every compared model. |
| `evaluation/bootstrap.py` | Performs paired bootstrap Pearson resampling, confidence intervals, and pairwise difference tests. |
| `evaluation/cli.py` | Provides the installed `debour-evaluate` command for already-generated prediction tables. |

### `src/debour/visualization/`: Figures

| File | Purpose |
|---|---|
| `visualization/__init__.py` | Exposes common performance plotting functions. |
| `visualization/performance.py` | Draws training-size learning curves and bootstrap confidence-interval plots with optional top-three highlighting. |
| `visualization/coverage.py` | Compares normalized facility coverage as additional sequences are selected. |
| `visualization/umap.py` | Fits/reuses UMAP embeddings and creates consistently styled full-data and selection-overlay facets. |

### `src/debour/synthetic/`: Controlled Redundancy

| File | Purpose |
|---|---|
| `synthetic/__init__.py` | Exposes prototype and mutation-family generation. |
| `synthetic/prototypes.py` | Selects diverse prototype sequences with greedy farthest-first weighted JS distance. |
| `synthetic/mutations.py` | Creates 50–100 or otherwise configured mutated copies per parent with known mutation counts. |
| `synthetic/oracle_labels.py` | Predicts synthetic labels with a trained BHI teacher and preserves the Table S2 target contract. |

### `src/debour/cluster/`: Slurm Support

| File | Purpose |
|---|---|
| `cluster/__init__.py` | Exposes Slurm resource and script-generation helpers. |
| `cluster/slurm.py` | Writes batch scripts with configurable CPU, memory, time, GPU, account, and optional partition settings. |
| `cluster/commands.py` | Shell-safely prints or explicitly executes argument-vector commands assembled by experiment runners. |

### `experiments/`: Historical Studies

| Folder | Purpose |
|---|---|
| `experiments/real_k562_active_learning/` | Reproduces the 120k base and repeated +20k uncertainty learning curve with shared-holdout evaluation. |
| `experiments/lsh_centroid_selection/` | Reproduces real-data LSH centroid facility and hybrid selection. |
| `experiments/synthetic_family_active_learning/` | Reproduces the known-family 10k base plus 2k acquisition benchmark. |
| `experiments/hybrid_lambda_sweep/` | Reproduces conditional/unconditional lambda sweeps over LSH or family centroids. |

### `configs/`: Human-Readable Experiment Settings

| File | Purpose |
|---|---|
| `configs/model/bhi_cnn_bilstm.yaml` | Canonical model architecture and training hyperparameters. |
| `configs/selection/random.yaml` | Random acquisition baseline settings. |
| `configs/selection/uncertainty.yaml` | Ensemble-variance acquisition settings. |
| `configs/selection/facility_unconditional.yaml` | Facility selection with zero initial coverage. |
| `configs/selection/facility_conditional.yaml` | Facility selection initialized from existing training-set coverage. |
| `configs/selection/hybrid.yaml` | Hybrid score definition, normalization choices, and uncertainty lambda. |
| `configs/selection/family_facility_unconditional.yaml` | Synthetic family-centroid facility selection with zero initial coverage. |
| `configs/selection/family_facility_conditional.yaml` | Synthetic family-centroid facility selection initialized from base-set coverage. |
| `configs/selection/family_hybrid.yaml` | Synthetic known-family hybrid selection and score normalization. |
| `configs/experiments/k562_active_learning.yaml` | Reference 120k-to-180k real-data active-learning experiment design. |
| `configs/experiments/synthetic_benchmark.yaml` | Reference prototype, mutation-family, base-size, and acquisition-budget settings. |

### `docs/`: Scientific And Operational Record

| File | Purpose |
|---|---|
| `docs/PIPELINE.md` | Technical explanation of data flow, architecture, objectives, and evaluation contracts. |
| `docs/COMMANDS.md` | Copyable local/Nibi commands for installation, training, selection, evaluation, and job inspection. |
| `docs/PROJECT_NOTES.md` | Chronological record of what was attempted, what worked, and what did not scale. |
| `docs/RESULTS.md` | Consolidated quantitative observations with provenance caveats. |
| `docs/DECISIONS.md` | Methodological decisions, rationale, limitations, and recommended controls. |
| `docs/SMOKE_TEST_REPORT.md` | Package-wide verification coverage, fixes applied, and remaining environment limitations. |
| `docs/PAPER_OUTLINE.tex` | Compile-ready LaTeX manuscript outline covering sections, equations, figures, tables, scope, and required analyses. |

### `tests/`: Fast Contract Checks

| File | Purpose |
|---|---|
| `tests/test_sequences.py` | Checks padding, exact cropping, channel order, and `N` encoding. |
| `tests/test_kmers.py` | Checks overlapping counts, 84-feature shape, and independent block normalization. |
| `tests/test_jensen_shannon.py` | Checks JS identity, symmetry, maximum separation, and weighted-distance behavior. |
| `tests/test_splits.py` | Checks deterministic/disjoint splits and leakage errors. |
| `tests/test_facility_location.py` | Checks that facility selection covers distinct representative regions. |
| `tests/test_hybrid_selection.py` | Checks that the uncertainty-only lambda endpoint selects maximum variance. |
| `tests/test_family_selection.py` | Checks known-family centroid means, remaining-family-size weights, stable labels, and metadata validation. |
| `tests/test_cluster_commands.py` | Checks shell-safe experiment command rendering and non-executing dry runs. |
| `tests/test_bootstrap.py` | Checks paired-bootstrap dimensions and expected correlation direction. |

The `__init__.py` files mark folders as importable Python packages and expose a
small set of convenient public functions. They contain no experiment-specific
data or hidden processing.

## Generated Artifact Map

Generated artifacts are written under the output directory supplied to a
workflow. They are ignored by Git by default because checkpoints and complete
prediction tables can be large.

### Training Run

| Artifact | Meaning |
|---|---|
| `shared_ids.txt` | Exact data pool shared by all ensemble members. |
| `seed42/`, `seed43/`, etc. | One independently trained model run per seed. |
| `splits/sampled_ids.txt` | Every ID available to that model run. |
| `splits/train_ids.txt` | IDs used for gradient updates. |
| `splits/val_ids.txt` | IDs used for validation and checkpoint selection. |
| `run_config.json` | Architecture, optimizer, loss, scheduler, seed, sizes, and parameter count. |
| `history.json` | Per-epoch training Huber loss, validation MSE/Pearson, and elapsed time. |
| `last.pt` | Final resumable checkpoint with model, optimizer, and scheduler states. |
| `model_best.pth` | Weights from the epoch with the highest validation Pearson r. |
| `summary.json` | Best epoch and best validation score plus run metadata. |

### Selection Run

| Artifact | Meaning |
|---|---|
| `all_candidate_variances.tsv.gz` | Every eligible row ordered with per-model predictions and ensemble variance. |
| `selection.tsv` | Selected rows in acquisition order with method-specific component scores. |
| `selected_ids.txt` | Ordered IDs to union with the base training set. |
| `selection_config.json` | Method, seeds, paths, candidate count, budget, normalization, and other settings. |
| `family_ground_summary.tsv` | Known parent-family labels, remaining-family-size weights, centroids, and conditional starting coverage when family selection is used. |

### Evaluation Run

| Artifact | Meaning |
|---|---|
| `predictions.tsv.gz` | Shared heldout rows and each compared model's prediction. |
| `bootstrap_summary.tsv` | Observed Pearson r, bootstrap mean, and 95% confidence interval per model. |
| `pairwise_tests.tsv` | Paired model differences and two-sided bootstrap p-values. |
| `bootstrap_distributions.tsv.gz` | Pearson r from every bootstrap replicate. |
| `bootstrap_pearson_summary.png` | Confidence-interval plot, with the top three models highlighted and no legend. |
| `evaluation_config.json` | Test IDs, seeds, model paths, bootstrap count, and inference settings. |

### Synthetic Benchmark Run

| Artifact | Meaning |
|---|---|
| `prototypes.tsv` | Diverse parent sequences selected by farthest-first traversal. |
| `mutated_copies.tsv.gz` | Synthetic IDs, parent IDs, mutation counts, and mutated sequences. |
| `synthetic_config.json` | Prototype count, family-size range, mutation levels, seed, and source table. |

## Install

From this directory:

```bash
module load python/3.11                # Compute Canada/Nibi
source ~/dream-rnn-env/bin/activate    # existing environment
python3 -m pip install -e .
```

On a network-restricted cluster, prevent `pip` from trying to download build
dependencies and use the modules already loaded in your environment:

```bash
python3 -m pip install -e . --no-build-isolation
```

If installation is unavailable, commands can still run directly from this
folder by prefixing them with `PYTHONPATH=src`, for example
`PYTHONPATH=src python3 workflows/train_ensemble.py --help`.

For UMAP and statistical tests:

```bash
python3 -m pip install -e '.[umap,stats]'
```

## Quick Start

Validate installation and run tests:

```bash
python3 -m pytest
```

Train a five-model ensemble from a sequence table:

```bash
python3 workflows/train_ensemble.py \
  --table ../data/DATA-Table_S2__MPRA_dataset.txt \
  --sample-rows 10000 \
  --seeds 42 43 44 45 46 \
  --output-dir outputs/example_ensemble
```

Score unused rows with an ensemble and select the 2,000 most uncertain:

```bash
python3 workflows/select_sequences.py uncertainty \
  --table ../data/DATA-Table_S2__MPRA_dataset.txt \
  --model-runs outputs/example_ensemble/seed42 outputs/example_ensemble/seed43 \
    outputs/example_ensemble/seed44 outputs/example_ensemble/seed45 outputs/example_ensemble/seed46 \
  --exclude-ids outputs/example_ensemble/shared_ids.txt \
  --select-rows 2000 \
  --output-dir outputs/uncertainty_2k
```

Evaluate checkpoints on one shared heldout set with paired bootstrap samples:

```bash
python3 workflows/evaluate_models.py \
  --table ../data/DATA-Table_S2__MPRA_dataset.txt \
  --model-runs outputs/example_ensemble/seed42 outputs/another_model \
  --test-rows 10000 --bootstrap-samples 1000 \
  --output-dir outputs/evaluation
```

Run LSH-centroid conditional facility selection from a normalized k-mer table:

```bash
python3 workflows/select_facility.py \
  --kmer-table outputs/kmer_counts.tsv \
  --base-ids outputs/example_ensemble/shared_ids.txt \
  --method facility --conditional \
  --select-rows 2000 \
  --output-dir outputs/conditional_facility_2k
```

Run the corresponding synthetic experiment with true parent families replacing
LSH buckets:

```bash
python3 workflows/select_family_facility.py \
  --synthetic-table outputs/synthetic_mut1_19/mutated_copies.tsv \
  --kmer-table outputs/synthetic_mut1_19/kmer_counts_normalized.tsv \
  --base-ids outputs/base10k/shared_ids.txt \
  --conditional --method facility --select-rows 2000 \
  --output-dir outputs/family_conditional_2k
```

The family workflow writes `selection.tsv`, `selected_ids.txt`,
`family_ground_summary.tsv`, and `selection_config.json`. Omit `--conditional`
for an unconditional diversity baseline. Add `--method hybrid`, a variance
table, and `--lambda-uncertainty` to combine family coverage with ensemble
disagreement.

## Design Guarantees

- Sequence encoding is `A,G,C,T`, matching the historical training code.
- Short sequences are left-padded with `N`; long sequences are exactly center-cropped.
- Ensemble uncertainty is sample variance across model predictions (`ddof=1`).
- All compared models use the same heldout IDs and paired bootstrap resamples.
- Conditional facility location accounts for coverage already supplied by the base training set.
- Selection outputs preserve rank, component scores, configuration, and selected IDs.
- Random seeds and source paths are persisted in every run configuration.

## Documentation

- [Pipeline](docs/PIPELINE.md): end-to-end data flow and interfaces.
- [Commands](docs/COMMANDS.md): local and Nibi examples.
- [Project notes](docs/PROJECT_NOTES.md): what was tried and what was learned.
- [Results](docs/RESULTS.md): consolidated empirical observations.
- [Decisions](docs/DECISIONS.md): methodological choices and caveats.
- [Smoke-test report](docs/SMOKE_TEST_REPORT.md): verification coverage, fixes, and limitations.

## Data Policy

Large tables and model checkpoints are not copied into this package. Commands accept paths to them, and `.gitignore` excludes common generated artifacts. Commit code, small configs, documentation, and compact result summaries; store large data/checkpoints in project storage or a release artifact system.
