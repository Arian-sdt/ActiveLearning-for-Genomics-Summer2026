# Smoke Test Report

## Purpose

Record the package-wide verification performed on August 27, 2026, including
the exact coverage, issues found, fixes applied, and environmental limitations.

## How To Use

Use this report as a baseline when changing the pipeline. Run
`python3 -m pytest` in an environment with the `dev` extra, then repeat the
workflow commands in `docs/COMMANDS.md` when changes affect orchestration,
training, inference, selection, or output formats.

## Environment

- Python 3.13 local CPU environment
- NumPy 2.1.2
- pandas 2.2.3
- PyTorch 2.11.0
- Matplotlib 3.10.3
- PyYAML 6.0.2
- `umap-learn` not installed locally
- `pytest` not installed locally

The 15 pytest test functions were therefore loaded and executed with a
minimal `pytest.raises` compatibility shim. This checks their assertions but is
not a substitute for running the real pytest test runner in CI or on Nibi.

## Static And Interface Checks

- All Python files compiled successfully with `compileall`.
- All 43 package modules imported successfully.
- All seven workflow scripts imported successfully.
- All four experiment runners returned `--help` successfully and completed
  representative dry-run command assembly without launching work.
- Every documented module and workflow CLI returned `--help` successfully.
- Public package-level convenience imports worked after conversion to lazy
  imports.
- The model self-check reported 6,807,489 parameters and the expected output
  shape.
- All 15 YAML files parsed into non-empty mappings.
- `pyproject.toml` parsed successfully.
- All relative Markdown links in the README and documentation resolved.

## Functional Module Checks

The following behavior was exercised with small temporary fixtures:

- Table schema customization and required-column contracts
- Chunk cleaning, numeric target conversion, invalid-sequence removal,
  duplicate removal, inclusion/exclusion filtering, and ID file I/O
- Sequence validation, left padding, exact center cropping, GC content, and
  `A,G,C,T` one-hot channel order
- Deterministic sampling/splitting and overlap detection
- 1/2/3-mer enumeration, overlapping counts, 84-feature shape, and independent
  probability normalization
- Jensen-Shannon identity, symmetry, bounded maximum, weighted distance, and
  weighted similarity
- CNN+BiLSTM forward pass
- Raw state-dictionary checkpoint discovery/loading
- Single-model and multi-model batched inference
- Ensemble mean, sample variance, and standard deviation
- Oracle labeling
- Random and uncertainty selection
- Selection table/config/ID persistence
- Conditional coverage initialization
- Lazy-greedy facility selection
- Hybrid selection and component normalization
- Random-hyperplane LSH and weighted bucket centroids
- Pearson, MSE, RMSE, and MAE
- Shared heldout construction and training-ID discovery
- Paired bootstrap summaries, distributions, and pairwise comparisons
- Learning-curve, bootstrap, coverage, and UMAP-facet file generation
- Prototype farthest-first selection
- Controlled mutation generation and family metadata
- Known-family centroid construction and remaining-family-size weighting
- Slurm batch-script generation

All checks passed.

## Existing Test Files

All 15 test functions in nine test files passed:

- `test_sequences.py`: 2
- `test_kmers.py`: 2
- `test_jensen_shannon.py`: 2
- `test_splits.py`: 2
- `test_facility_location.py`: 1
- `test_hybrid_selection.py`: 1
- `test_family_selection.py`: 2
- `test_cluster_commands.py`: 2
- `test_bootstrap.py`: 1

## End-To-End Workflow Checks

A temporary 80-row, 200 bp sequence/activity table was used to run the complete
command chain on CPU:

1. `train_ensemble.py`: trained two independent one-epoch models on one shared
   40-row set and wrote best/last checkpoints, histories, configs, summaries,
   and split IDs.
2. `select_sequences.py random`: selected and saved three random unused rows.
3. `select_sequences.py uncertainty`: scored all 40 remaining rows with both
   checkpoints, saved every variance, and selected the top three.
4. `evaluate_models.py`: evaluated both models on one shared 10-row holdout with
   20 paired bootstrap samples and generated all result tables and the PNG.
5. Normalized k-mer CLI: generated the complete 84-feature table.
6. `select_facility.py`: completed both conditional and unconditional
   LSH-centroid facility selection.
7. `select_facility.py` hybrid: combined candidate variance and conditional
   facility gain and saved two selected rows with diagnostics.
8. `run_active_learning_round.py`: scored, selected two additions, combined IDs,
   and trained a fresh one-epoch expanded model.
9. `run_synthetic_benchmark.py`: selected four farthest-first prototypes and
   generated controlled 1/4-mutation families.
10. Installed-style selection/evaluation and direct sequence/split/JS module
    CLIs all completed and wrote their expected files.
11. `select_family_facility.py`: replaced LSH with known parent families and
    completed unconditional facility, conditional facility, and conditional
    hybrid selection. Every mode saved ordered selections, IDs, family ground
    summaries, and reproducibility configuration.

All end-to-end checks passed.

## Issues Found And Fixed

1. **Headless plotting abort:** Matplotlib selected an interactive native backend
   in a display-free process. All plotting modules now explicitly use `Agg`, the
   correct backend for Nibi batch jobs.
2. **`python -m` runtime warnings:** eager imports in four `__init__.py` files
   preloaded the module being executed. Public APIs now use lazy imports.
3. **Missing best checkpoint for undefined Pearson:** a one-row or constant
   validation set can produce `NaN` Pearson and previously leave no
   `model_best.pth`. Epoch one is now always saved as a fallback; later finite
   improvements replace it normally.
4. **Non-standard JSON `NaN`:** undefined metrics previously serialized as bare
   `NaN`. Training history and summaries now write strict JSON `null` values.

## Remaining Limitations

- `umap-learn` was unavailable, so UMAP fitting was not executed. The UMAP facet
  plotting function was executed with a synthetic embedding, and the missing
  optional dependency produced the intended installation message.
- GPU/CUDA execution was not available locally. CPU training, checkpointing, and
  inference use the same model/data paths, but CUDA availability and Nibi module
  compatibility must still be checked in the target environment.
- Full-scale Table S2 runtime and memory behavior were not rerun. Smoke tests
  validate correctness and interfaces, not production-scale completion time.
- Slurm scripts were generated and inspected but not submitted from this local
  machine.
