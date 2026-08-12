#ifndef TREE_HMM_SRC_CUDA_DEVICE_ALGEBRA_H_
#define TREE_HMM_SRC_CUDA_DEVICE_ALGEBRA_H_

#include <cmath>

#if defined(__CUDACC__)
#define TREE_HMM_CUDA_HOST_DEVICE __host__ __device__
#else
#define TREE_HMM_CUDA_HOST_DEVICE
#endif

namespace tree_hmm::cuda::detail {

TREE_HMM_CUDA_HOST_DEVICE inline float
RakeValue(const float *path, const float *node, unsigned states,
          unsigned parent_state) {
  float result = 0.0f;
  for (unsigned child_state = 0; child_state < states; ++child_state)
    result += path[parent_state * states + child_state] * node[child_state];
  return result;
}

TREE_HMM_CUDA_HOST_DEVICE inline float
CompressionValue(const float *left, const float *middle, const float *right,
                 unsigned states, unsigned parent_state,
                 unsigned child_state) {
  float result = 0.0f;
  for (unsigned middle_state = 0; middle_state < states; ++middle_state) {
    result += left[parent_state * states + middle_state] *
              middle[middle_state] *
              right[middle_state * states + child_state];
  }
  return result;
}

TREE_HMM_CUDA_HOST_DEVICE inline float Product(float left, float right) {
  return left * right;
}

TREE_HMM_CUDA_HOST_DEVICE inline float Maximum(const float *values,
                                               unsigned size) {
  float result = 0.0f;
  for (unsigned index = 0; index < size; ++index)
    result = fmaxf(result, values[index]);
  return result;
}

TREE_HMM_CUDA_HOST_DEVICE inline float UpdatedLogScale(float input_scale,
                                                       float maximum) {
  return maximum > 0.0f ? input_scale + logf(maximum) : -INFINITY;
}

} // namespace tree_hmm::cuda::detail

#undef TREE_HMM_CUDA_HOST_DEVICE

#endif // TREE_HMM_SRC_CUDA_DEVICE_ALGEBRA_H_
