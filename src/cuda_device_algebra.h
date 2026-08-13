#ifndef TREE_HMM_SRC_CUDA_DEVICE_ALGEBRA_H_
#define TREE_HMM_SRC_CUDA_DEVICE_ALGEBRA_H_

#include <cmath>

#include "tree_hmm/scalar.h"

#if defined(__CUDACC__)
#define TREE_HMM_CUDA_HOST_DEVICE __host__ __device__
#else
#define TREE_HMM_CUDA_HOST_DEVICE
#endif

namespace tree_hmm::cuda::detail {

TREE_HMM_CUDA_HOST_DEVICE inline Scalar RakeValue(const Scalar *path,
                                                  const Scalar *node,
                                                  unsigned states,
                                                  unsigned parent_state) {
  Scalar result = 0.0f;
  for (unsigned child_state = 0; child_state < states; ++child_state)
    result += path[parent_state * states + child_state] * node[child_state];
  return result;
}

TREE_HMM_CUDA_HOST_DEVICE inline Scalar
MatrixProductValue(const Scalar *left, const Scalar *right, unsigned states,
                   unsigned parent_state, unsigned child_state) {
  Scalar result = 0.0f;
  for (unsigned middle_state = 0; middle_state < states; ++middle_state) {
    result += left[parent_state * states + middle_state] *
              right[middle_state * states + child_state];
  }
  return result;
}

TREE_HMM_CUDA_HOST_DEVICE inline Scalar Product(Scalar left, Scalar right) {
  return left * right;
}

TREE_HMM_CUDA_HOST_DEVICE inline Scalar Maximum(const Scalar *values,
                                                unsigned size) {
  Scalar result = 0.0f;
  for (unsigned index = 0; index < size; ++index)
    result = fmax(result, values[index]);
  return result;
}

TREE_HMM_CUDA_HOST_DEVICE inline Scalar UpdatedLogScale(Scalar input_scale,
                                                        Scalar maximum) {
  return maximum > 0.0f ? input_scale + log(maximum) : -INFINITY;
}

} // namespace tree_hmm::cuda::detail

#undef TREE_HMM_CUDA_HOST_DEVICE

#endif // TREE_HMM_SRC_CUDA_DEVICE_ALGEBRA_H_
