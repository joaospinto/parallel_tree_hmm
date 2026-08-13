#ifndef TREE_HMM_ROCM_H_
#define TREE_HMM_ROCM_H_

#include <cstddef>
#include <memory>
#include <string>

#include "tree_hmm/accelerator.h"

namespace tree_hmm::rocm {

#include "tree_hmm/detail/gpu_backend_api.inc"

} // namespace tree_hmm::rocm

#endif // TREE_HMM_ROCM_H_
