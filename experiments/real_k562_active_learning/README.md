# Real K562 Active Learning

## Purpose

Reproduce the original real-data experiment: train five 120k models, rank unused
rows by ensemble variance, add 20k per round, retrain from scratch at 140k/160k/
180k, and compare seed-42 checkpoints on one unseen heldout set. Random nested
20k/40k/60k additions are the size-matched baseline.

## How To Run

Use `run.py train-base`, then `run.py uncertainty-round` once per round, and
finally `run.py evaluate`. Commands print by default; add `--execute` before the
stage after checking paths. See `python3 run.py --help`.

## Historical Outcome

Heldout Pearson improved with active additions in the recorded seed-42 curve,
but gains diminished with each 20k increment. Fair conclusions require the same
heldout IDs for random and uncertainty models and should be repeated across
training seeds rather than inferred from seed 42 alone.
