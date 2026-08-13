#include "tree_hmm/cuda.h"

#include <stdexcept>

#define TREE_HMM_GPU_STUB_NAMESPACE cuda
#define TREE_HMM_GPU_STUB_DESCRIPTION "CUDA backend not built"
#define TREE_HMM_GPU_STUB_ERROR                                                \
  "CUDA backend not built; use Bazel --config=cuda and link tree_hmm_cuda"
#include "src/gpu_backend_stub.inc"
