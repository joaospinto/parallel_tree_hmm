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
- batched CUDA, ROCm, and Metal likelihood, posterior-marginal,
  MAP-reconstruction, and posterior-sampling backends sharing one accelerator
  API;
- brute-force, host-side device-algebra, and accelerator cross-validation.

The CPU, CUDA, and ROCm implementations use one compile-time
`tree_hmm::Scalar`. FP64 is the default; FP32 is a separate pure-precision
build. Likelihoods use scale propagation and marginals use a log-domain
contraction, so neither underflows on large trees. Metal is FP32-only. Prepared
CPU and accelerator APIs allocate all problem storage in the corresponding
workspace reservation; repeated numerical calls reuse it.

MAP assignments and posterior samples use the same topology plan as sum-product
inference. Contraction records the local conditional data required for
reconstruction, and reverse traversal restores all eliminated states in
dependency order. CPU prepared calls return workspace-backed views and allocate
no memory. The accelerator APIs provide the same marginal, MAP, and
posterior-sampling behavior for batches. Accelerator operation-specific
workspace reservations own only the choice tapes, conditional-factor tapes, or
reverse adjoints required by the requested result, so likelihood-only
accelerator workspaces do not pay recovery storage costs. Posterior sampling
accepts one `U[0,1)` variate per node so random-number generation remains
external, reproducible, and independent of execution order. Posterior-marginal
calls return workspace-backed batch-major node and edge probabilities together
with one log partition function per batch item.

CUDA, ROCm, and Metal pack independent operations into full threadgroups for
small state spaces, with a generic-state fallback. Each accelerator workspace
also exposes a mutable model view through `Inputs()`. Applications can prepare
factors directly in the workspace's pinned CUDA/ROCm storage or shared Metal
storage, then pass the resulting ordinary `BatchedModelView` to the same
inference call. This avoids a full-batch staging copy without introducing a
separate numerical path. All three backends also expose `CategoricalInputs()` for
models in which selected nodes carry byte-valued observations. Those inputs
remain compact: an initialization kernel combines the observations with a
shared emission table directly in the inference workspace instead of
materializing a batch of dense node factors on the host.

Categorical prepared calls accept a `CategoricalInputUpdate` policy. The
default, `kAll`, preserves the ordinary stateless-call behavior. After one
`kAll` call, `kFactors` leaves the observations resident while explicitly
updating the root, emission, and edge factors; this matches repeated
phylogenetic likelihood evaluations with fixed tip data and changing model or
branch parameters. `kNone` reuses both observations and factors for
fixed-model throughput measurements. The workspace rejects either reuse mode
until the required data have been staged, rejects observation reuse at a
different batch size, and invalidates residency whenever it is reserved
again. The returned timings distinguish requested host-to-device updates,
accelerator execution, result download, and full prepared-call latency.

CUDA and ROCm share one kernel and execution implementation over the common
CUDA/HIP runtime subset; thin wrappers select the public namespace and runtime.
The shared CUDA/HIP device algebra is also compiled and exercised as ordinary
host code on every build. This checks its numerical operations but does not
emulate GPU execution. Real CUDA or ROCm compilation and launch semantics
require the corresponding SDK, and performance validation requires the
corresponding GPU. The Metal kernels are compiled and tested directly on
macOS.

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

ROCm compilation is opt-in. It can be validated without an AMD GPU; execution
of the test requires one. The default target is MI300 (`gfx942`) and can be
changed with `TREE_HMM_ROCM_ARCH`:

```sh
scripts/rocm.sh build
TREE_HMM_ROCM_ARCH=gfx90a scripts/rocm.sh build
scripts/rocm.sh test
```

On Linux, Bazel fetches a pinned ROCm 7.2.3 SDK when `ROCM_PATH` is unset.
Set `ROCM_PATH` to use an existing installation instead. CUDA targets likewise
use a pinned CUDA 12.8.1 redistribution, so CUDA and ROCm compilation do not
depend on a system toolkit. A corresponding GPU and driver are still required
to execute either backend.

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
