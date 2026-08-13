#ifndef TREE_HMM_SRC_ROCM_RUNTIME_H_
#define TREE_HMM_SRC_ROCM_RUNTIME_H_

// ROCm 7.2.3's public HIP headers define unused helper functions and retain an
// unused kernel parameter. Suppress those two diagnostics only while parsing
// the pinned third-party headers; project and shared CUDA/HIP sources continue
// to compile with -Werror for every warning category.
#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-function"
#pragma clang diagnostic ignored "-Wunused-parameter"
#endif

#include <hip/hip_runtime.h>

#if defined(__clang__)
#pragma clang diagnostic pop
#endif

#endif // TREE_HMM_SRC_ROCM_RUNTIME_H_
