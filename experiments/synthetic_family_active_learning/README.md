# Synthetic Family Active Learning

## Purpose

Provide a controlled redundancy benchmark. Diverse real K562 prototypes become
parents of mutation families. A full-table BHI teacher supplies synthetic oracle
labels. Five models share one random 10k base, then fresh 12k models compare
random, uncertainty, unconditional diversity, conditional diversity, and hybrid
2k additions.

## How To Run

Run `generate`, label the resulting table with the reusable oracle code, run
`train-base`, produce ensemble variances, and then run the desired selection
stages. Commands print unless `--execute` appears before the stage. Use
`--training-ids` with `train-base` to train on an already combined 12k ID set.

## Why Families Replace LSH

Every synthetic row records its true `parent_id`, so approximate hash buckets are
unnecessary. The selector excludes base IDs, computes one centroid per remaining
family, weights it by remaining family size, and optionally initializes coverage
from the base 10k.

## Historical Outcome

This benchmark made redundancy explicit and enabled direct method comparisons.
Row-random holdouts can leak parent-family information, so final claims should
also use family-heldout evaluation where no parent's copies cross train/test.
