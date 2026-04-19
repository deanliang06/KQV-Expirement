## Truncated SVD Baseline

This experiment provides a non-learned baseline for the `Q` replacement:

- take the original `Q` matrix from each GPT attention block
- compute a rank-`r` truncated SVD approximation
- use that approximation in place of the learned `Q` path

`K` and `V` remain unchanged so the hook point matches the U-Net experiments.

Use this to compare the learned bottleneck approach against a fixed low-rank baseline with the same effective bottleneck size.
