# Average-Then-Select Mean-Std Summaries

These tables aggregate raw validation curves over `seed0`, `seed1`, `seed2` before selecting
the reported checkpoint.

For each method, the pipeline computes the per-step arithmetic mean and sample
standard deviation (`ddof=1`) for all metrics, using shared validation steps in
`[0, 150]`. It then selects one step by maximizing mean success rate,
breaking ties with mean task/test score and then the earliest step. Every metric
is reported at that selected step. `Base Model` is fixed at step 0.

- CSV and Markdown metric cells use `avg ± std`.
- LaTeX metric cells use `$avg \pm std$`.
- `Best Step` is a single integer selected from the cross-seed mean curve.
- Percentage metrics use one decimal place; ALFWorld `TestScore` uses up to
  three decimal places.
- Standard deviations are sample standard deviations over the three env seeds.

This differs from `results/combine_summary`, which selects each seed's best
checkpoint first and only then averages the three selected results.
