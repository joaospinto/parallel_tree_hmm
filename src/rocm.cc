#include "tree_hmm/rocm.h"

#include "src/rocm_runtime.h"
#include "src/hip_cuda_runtime_compat.h"

#define TREE_HMM_GPU_BACKEND_NAMESPACE rocm
#define TREE_HMM_GPU_BACKEND_ROCM 1
#include "src/gpu_backend_impl.inc"
