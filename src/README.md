# Reusable Source Code

## Purpose

Everything under `src/debour/` is experiment-independent library code. Files in
this directory define data contracts, feature calculations, models, training,
inference, acquisition functions, evaluation, plotting, synthetic generation,
and cluster helpers. They must not contain paths or conclusions belonging to one
particular experiment.

The corresponding experiment-specific layer lives in `../experiments/`. Those
scripts choose concrete stages, dataset sizes, seeds, paths, and comparisons,
then call this source package or the thin scripts in `../workflows/`.

## How To Use

Install the package from the repository root:

```bash
python3 -m pip install -e . --no-build-isolation
```

Then import reusable functions, for example:

```python
from debour.features.kmers import kmer_feature_matrix
from debour.selection.facility_location import greedy_facility_location
```

When editable installation is unavailable, prefix commands with
`PYTHONPATH=src`.

## Boundary Rule

- Put reusable algorithms and data transformations in `src/debour/`.
- Put reusable end-to-end command composition in `workflows/`.
- Put one study's paths, stages, parameters, and interpretation in
  `experiments/<experiment_name>/`.
- Put generated data, checkpoints, and figures outside the tracked source tree.
