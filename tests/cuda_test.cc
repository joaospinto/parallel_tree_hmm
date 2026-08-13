#include "tree_hmm/cuda.h"

#include "accelerator_test.h"

#define TREE_HMM_GPU_TEST_NAMESPACE cuda
#define TREE_HMM_GPU_TEST_NAME "CUDA"
#include "gpu_backend_test.inc"
