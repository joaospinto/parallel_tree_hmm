#include "tree_hmm/cuda.h"

#include <stdexcept>
#include <string>

int main() {
  if (tree_hmm::cuda::Available())
    throw std::runtime_error("the CPU CUDA stub reported an available device");
  if (tree_hmm::cuda::DeviceDescription().find("not built") ==
      std::string::npos) {
    throw std::runtime_error("the CPU CUDA stub description is unclear");
  }
  return 0;
}
