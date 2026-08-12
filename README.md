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
- stable scaled log-partition evaluation for long trees;
- batched Metal and CUDA likelihood backends sharing one accelerator API;
- brute-force, device-emulation, and accelerator cross-validation.

The CPU reference uses nonnegative double-precision factors. Accelerator
likelihoods use single precision with scale propagation attached to every
intermediate node, edge, and branch factor, so the returned log partitions do
not underflow on large trees. Prepared CPU and accelerator APIs allocate all
problem storage in `Workspace::Reserve`; repeated numerical calls reuse it.

The CUDA device algebra is exercised on every host build by an emulation test.
Real CUDA compilation, launch semantics, and performance still require an
NVIDIA system. The Metal kernels are compiled and tested directly on macOS.
Max-product recovery, posterior sampling, accelerator marginals, and language
bindings are later milestones.

## Build

Keep this repository beside `bidirectional_tree_rake_compress`, then run:

```sh
bazel test //...
```

On an NVIDIA host, build the real CUDA test with:

```sh
bazel test --config=cuda //:cuda_test
```

## License

MIT
