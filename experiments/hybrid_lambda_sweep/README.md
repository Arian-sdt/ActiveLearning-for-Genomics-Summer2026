# Hybrid Lambda Sweep

## Purpose

Preserve experiments combining ensemble uncertainty and facility-location gain:

`score(x) = lambda * normalized_variance(x) + (1-lambda) * normalized_facility_gain(x)`.

The same runner supports conditional/unconditional objectives and either LSH
centroids or true synthetic parent-family centroids.

## How To Run

Pass `--representation lsh` or `family`, required tables, and optional lambda
values to `run.py`. It prints all commands by default. Add `--execute` to run the
commands sequentially. For expensive runs, use the printed commands as payloads
for separate Slurm jobs instead.

## Historical Outcome

Min-max variance among high-uncertainty candidates often saturated near one,
while average facility marginal gain collapsed toward zero after early rounds.
Consequently even small positive lambdas could visually resemble pure
uncertainty. Rank-normalized experiments balanced scales differently but required
dynamic reranking and were substantially slower. Lambda comparisons must always
record both component scores, not only the final selected IDs.
