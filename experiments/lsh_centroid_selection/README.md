# LSH Centroid Selection

## Purpose

Preserve the real-data approximation used when exact all-pairs facility location
was too expensive. Random-hyperplane LSH groups normalized 1/2/3-mer vectors;
each occupied bucket becomes a centroid weighted by bucket size. Actual sequence
rows, not centroids, are selected.

## How To Run

Use one of the four stages in `run.py`: `facility-unconditional`,
`facility-conditional`, `hybrid-unconditional`, or `hybrid-conditional`.
Commands print by default. Add `--execute` before the stage to launch one.

## Historical Outcome

The 22-bit run produced far fewer occupied buckets than the theoretical hash
space because k-mer vectors are highly correlated. Within-bucket weighted JS
distance was materially below random same-size buckets, supporting LSH as a
cohesion approximation. Exact facility scoring over all sequences remained too
slow; weighted centroids made selection tractable but approximate.
