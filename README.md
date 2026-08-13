# Parallel Tree HMM

Discrete sum-product and posterior inference on arbitrary hidden Markov trees.
This repository owns the numerical algebras and public inference API; topology
planning and traversal come from the separate
[`bidirectional_tree_rake_compress`](https://github.com/joaospinto/bidirectional_tree_rake_compress)
package.

The package provides:

- a dense finite-state model with node and directed-edge potentials;
- a sum-product contraction for the partition function;
- stable log-domain reverse contraction for every node and edge posterior
  marginal, including models with zero factors;
- log-domain max-product contraction followed by MAP-state reconstruction;
- exact posterior sampling from caller-supplied uniform variates;
- stable scaled log-partition evaluation for long trees;
- batched Metal and CUDA likelihood, posterior-marginal, MAP-reconstruction,
  and posterior-sampling backends sharing one accelerator API;
- brute-force, device-emulation, and accelerator cross-validation.

The CPU and CUDA implementations use one compile-time `tree_hmm::Scalar`.
FP64 is the default; FP32 is a separate pure-precision build. Likelihoods use
scale propagation and marginals use a log-domain contraction, so neither
underflows on large trees. Metal is FP32-only. Prepared CPU and accelerator
APIs allocate all problem storage in the corresponding workspace reservation;
repeated numerical calls reuse it.

MAP assignments and posterior samples use the same topology plan as sum-product
inference. Contraction records the local conditional data required for
reconstruction, and reverse traversal restores all eliminated states in
dependency order. CPU prepared calls return workspace-backed views and allocate
no memory. The CUDA and Metal APIs provide the same marginal, MAP, and
posterior-sampling behavior for batches. Accelerator operation-specific
workspace reservations own only the choice tapes, conditional-factor tapes, or
reverse adjoints required by the requested result, so likelihood-only
accelerator workspaces do not pay recovery storage costs. Posterior sampling
accepts one `U[0,1)` variate per node so random-number generation remains
external, reproducible, and independent of execution order. Posterior-marginal
calls return workspace-backed batch-major node and edge probabilities together
with one log partition function per batch item.

CUDA and Metal pack independent operations into full threadgroups for small
state spaces, with a generic-state fallback. Each accelerator workspace also
exposes a mutable model view through `Inputs()`. Applications can prepare
factors directly in the workspace's pinned CUDA storage or shared Metal
storage, then pass the resulting ordinary `BatchedModelView` to the same
inference call. This avoids a full-batch staging copy without introducing a
separate numerical path. Both backends also expose `CategoricalInputs()` for
models in which selected nodes carry byte-valued observations. Those inputs
remain compact: an initialization kernel combines the observations with a
shared emission table directly in the inference workspace instead of
materializing a batch of dense node factors on the host.

The CUDA device algebra is exercised on every host build by an emulation test.
Real CUDA compilation, launch semantics, and performance still require an
NVIDIA system. The Metal kernels are compiled and tested directly on macOS.

## Build

Keep this repository beside `bidirectional_tree_rake_compress`. Build and test
the default FP64 CPU implementation with:

```sh
bazel test //... --config=fp64
```

On an NVIDIA host, build the real CUDA tests in the same precision:

```sh
bazel test //:cuda_test --config=fp64 --config=cuda
bazel test //:cuda_test --config=fp32 --config=cuda
```

Metal uses FP32:

```sh
bazel test //:metal_test --config=fp32
```

A dependent Bazel repository selects the precision explicitly with
`--@parallel_tree_hmm//:precision=fp64` or
`--@parallel_tree_hmm//:precision=fp32`. The selected type is part of the C++
ABI, so a binary and all of its tree-HMM dependencies must use the same build
configuration.

## License

MIT
