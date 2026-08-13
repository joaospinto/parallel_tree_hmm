#include "tree_hmm/cuda.h"
#include "tree_hmm/rocm.h"

#include <type_traits>

static_assert(!std::is_same_v<tree_hmm::cuda::Workspace,
                              tree_hmm::rocm::Workspace>);

int main() {
  tree_hmm::cuda::Workspace cuda_workspace;
  tree_hmm::rocm::Workspace rocm_workspace;
  return tree_hmm::cuda::Available() || tree_hmm::rocm::Available();
}
