# Results

## Purpose

Consolidate the main quantitative observations without mixing validation metrics, different heldout samples, or local and remote artifacts.

## How To Use

Treat these as a project record, not a frozen manuscript table. Regenerate final numbers from saved IDs/checkpoints using `workflows/evaluate_models.py` before publication.

## Real K562 Active Learning

One recorded seed-42 shared-holdout evaluation produced:

| Training rows | Selection history | Pearson r |
|---:|---|---:|
| 120,000 | initial random base | 0.83133 |
| 140,000 | base + uncertainty 20k | 0.84137 |
| 160,000 | base + two uncertainty rounds | 0.84823 |
| 180,000 | base + three uncertainty rounds | 0.85176 |

This is monotonic improvement with diminishing gains. A separate earlier heldout artifact reported approximately `0.83768`, `0.84740`, and `0.85344` for 120k/140k/160k. Those values must not be merged because the test sample or script version differed.

Local early experiments in this checkout compared 10k against 10k+100 uncertain sequences across several 1,000-row holdouts. Pearson changes ranged from nearly zero to about `+0.013`, generally favoring the expanded set but showing substantial small-test-set variability.

## LSH Validation

For 678,064 non-base rows:

| Hash bits | Occupied buckets |
|---:|---:|
| 14 | 370 |
| 18 | 1,124 |
| 22 | 1,511 |

With 22-bit LSH, mean within-bucket weighted JS distance was approximately `0.235154`; random buckets with identical sizes averaged `0.368309`. A very small Mann-Whitney p-value indicated that this difference was unlikely under equal distributions. More importantly, the effect size directly showed that LSH grouped more compositionally similar sequences.

## Facility-Location Runtime

Exact on-demand facility selection over roughly 100k candidates remained impractical. One initial scoring pass projected hours before any substantial 20k greedy sequence was selected. Loading k-mer tables once helped I/O but did not remove the dominant candidate-by-ground-by-round distance cost. Weighted LSH/family centroids made the objective tractable.

## Hybrid Selection

Ground-size-normalized facility marginal gains often became exactly or nearly zero after early selections, while min-max normalized variance among selected points was near one. One recorded 2k selection summary had mean normalized variance `0.998010` and median normalized facility gain `0`. Therefore even `lambda=0.01` visually resembled pure uncertainty in UMAP. Rank normalization improved component balance conceptually but required reranking changing facility gains each round and was much slower.

## Synthetic Benchmark

The controlled benchmark used 1,000 farthest-first K562 prototypes and 50–100 mutated copies per parent. One generated pool contained 75,560 sequences. Mutation schedules included 1–5, `1,4,7,...,19`, and increments up to 50. A full-Table-S2 BHI teacher supplied synthetic `K562_log2FC` labels; negative labels are expected because this regression output is unconstrained log2 fold change.

Experiments trained five 10k models, selected 2k additions by random, uncertainty, family-centroid facility, conditional family-centroid facility, and hybrid variants, then trained fresh 12k models. Final comparisons should be rerun from checkpoints on a single unseen heldout family-aware split to avoid related copies crossing train/test boundaries.

