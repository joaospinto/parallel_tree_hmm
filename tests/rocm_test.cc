#include "tree_hmm/rocm.h"

#include "accelerator_test.h"

#define TREE_HMM_GPU_TEST_NAMESPACE rocm
#define TREE_HMM_GPU_TEST_NAME "ROCm"
#include "gpu_backend_test.inc"
