# Decisions And Caveats

## Purpose

Record why the pipeline uses its current methods and which tempting alternatives were rejected or constrained.

## How To Use

Consult this before changing objectives or interpreting figures. Add a dated entry whenever an experiment changes the scientific contract.

## Decisions

1. **Use explicit IDs, not row positions.** Table order and duplicate construct rows make positions fragile.
2. **Use one shared pool for ensemble members.** Otherwise disagreement confounds data-set differences with epistemic model uncertainty.
3. **Use sample variance (`ddof=1`).** Five models estimate an ensemble disagreement distribution rather than a complete population.
4. **Train expanded models from scratch.** This keeps comparisons about selected data, not fine-tuning history.
5. **Use conditional facility location for additions to an existing set.** Unconditional facility location is still useful as a diversity baseline but ignores prior coverage.
6. **Use weighted centroids for scale.** Exact all-pair facility location is computationally prohibitive; bucket size preserves represented mass.
7. **Use family centroids for synthetic data.** Known parents are more meaningful than rebuilding approximate LSH groups.
8. **Keep raw component scores.** Hybrid normalization can dominate behavior in non-obvious ways; diagnostics must remain inspectable.
9. **Use shared paired bootstrap samples.** Independent resampling wastes pairing and weakens model-difference inference.
10. **Reuse UMAP embeddings for restyling only.** Refit when the feature pool or UMAP parameters change, not when colors/point size change.

## Caveats

- K-mer JS captures composition/local patterns, not alignment, motif position, or model-label behavior.
- LSH bit count does not imply `2^bits` occupied buckets; correlated 84-dimensional distributions occupy a small subset.
- Centroid similarity is an approximation to sequence-level coverage.
- Conventional UMAP geometry should not be used as a quantitative distance test.
- Tiny p-values can reflect very large sample sizes; report effect sizes and distributions.
- Oracle labels test recovery of a teacher's function, not biological truth.
- Synthetic copies from one parent are related. Random row-level test splits can leak family information; family-heldout evaluation is preferable.
- Pearson r measures ranking/linear association, not calibration. Report MSE/MAE too.
- The 30 bp flanking-context field described in the source model table was not reconstructable from the provided Table S2 sequence column.

