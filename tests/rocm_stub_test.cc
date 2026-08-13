#include "tree_hmm/rocm.h"

#include <stdexcept>
#include <string>

int main() {
  if (tree_hmm::rocm::Available())
    throw std::runtime_error("the CPU ROCm stub reported an available device");
  if (tree_hmm::rocm::DeviceDescription().find("not built") ==
      std::string::npos) {
    throw std::runtime_error("the CPU ROCm stub description is unclear");
  }
  return 0;
}
