# Parallel Tree HMM

Discrete sum-product and posterior inference on arbitrary hidden Markov trees.
This repository owns the numerical algebras and public inference API; topology
planning and traversal come from the separate
[`bidirectional_tree_rake_compress`](../bidirectional_tree_rake_compress)
package.

The first vertical slice provides:

- a dense finite-state model with node and directed-edge potentials;
- a sum-product contraction for the partition function;
- stable log-domain reverse contraction for every node and edge posterior
  marginal, including models with zero factors;
- log-domain max-product contraction followed by MAP-state reconstruction;
- exact posterior sampling from caller-supplied uniform variates;
- stable scaled log-partition evaluation for long trees;
- batched Metal and CUDA likelihood and MAP-reconstruction backends sharing
  one accelerator API;
- brute-force, device-emulation, and accelerator cross-validation.

The CPU implementation uses nonnegative double-precision factors. Likelihoods
use scale propagation and marginals use a log-domain contraction, so neither
underflows on large trees. Accelerator likelihoods use single precision with
scale propagation attached to every intermediate node, edge, and branch
factor. Prepared CPU and accelerator APIs allocate all problem storage in
`Workspace::Reserve`; repeated numerical calls reuse it.

MAP assignments and posterior samples use the same topology plan as sum-product
inference. Contraction records the local conditional data required for
reconstruction, and reverse traversal restores all eliminated states in
dependency order. CPU prepared calls return workspace-backed views and allocate
no memory. The CUDA and Metal MAP APIs provide the same behavior for batches;
their explicitly bidirectional workspace reservations own the additional
choice tapes and recovered assignments, so likelihood-only workspaces do not
pay that storage cost. Posterior sampling accepts one `U[0,1)` variate per node
so random-number generation remains external, reproducible, and independent of
execution order.

CUDA and Metal pack independent operations into full threadgroups for small
state spaces, with a generic-state fallback. Each accelerator workspace also
exposes a mutable model view through `Inputs()`. Applications can prepare
factors directly in the workspace's pinned CUDA storage or shared Metal
storage, then pass the resulting ordinary `BatchedModelView` to the same
inference call. This avoids a full-batch staging copy without introducing a
separate numerical path.

The CUDA device algebra is exercised on every host build by an emulation test.
Real CUDA compilation, launch semantics, and performance still require an
NVIDIA system. The Metal kernels are compiled and tested directly on macOS.
Accelerator marginals, posterior sampling, and language bindings are later
milestones.

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
