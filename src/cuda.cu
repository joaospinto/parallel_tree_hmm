#include "tree_hmm/cuda.h"

#include <cuda_runtime.h>

#define TREE_HMM_GPU_BACKEND_NAMESPACE cuda
#define TREE_HMM_GPU_BACKEND_CUDA 1
#include "src/gpu_backend_impl.inc"
