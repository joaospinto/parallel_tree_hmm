#include "tree_hmm/rocm.h"

#include <stdexcept>

#define TREE_HMM_GPU_STUB_NAMESPACE rocm
#define TREE_HMM_GPU_STUB_DESCRIPTION "ROCm backend not built"
#define TREE_HMM_GPU_STUB_ERROR                                                \
  "ROCm backend not built; use Bazel --config=rocm and link tree_hmm_rocm"
#include "src/gpu_backend_stub.inc"
