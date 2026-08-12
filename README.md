# Parallel Tree HMM

Discrete sum-product, max-product, and posterior inference on arbitrary hidden
Markov trees. This repository owns the numerical algebras and public inference
API; topology planning and traversal come from the separate
[`bidirectional_tree_rake_compress`](../bidirectional_tree_rake_compress)
package.

The first vertical slice provides:

- a dense finite-state model with node and directed-edge potentials;
- a sum-product contraction for the partition function;
- analytic reverse contraction for every node and edge posterior marginal;
- brute-force cross-validation on a branching, unbalanced tree.

The reference implementation uses nonnegative double-precision factors. The
accelerator implementation will add explicit normalization/scaling before
large biological problems are benchmarked. Max-product recovery, posterior
sampling, batches, CUDA, Metal, and language bindings are the next milestones.

## Build

Keep this repository beside `bidirectional_tree_rake_compress`, then run:

```sh
bazel test //...
```

## License

MIT
