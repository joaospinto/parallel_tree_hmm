#ifndef TREE_HMM_CUDA_H_
#define TREE_HMM_CUDA_H_

#include <cstddef>
#include <memory>
#include <string>

#include "tree_hmm/accelerator.h"

namespace tree_hmm::cuda {

bool Available();
std::string DeviceDescription(int device = 0);

class Workspace {
public:
  struct Impl;

  Workspace();
  ~Workspace();
  Workspace(Workspace &&) noexcept;
  Workspace &operator=(Workspace &&) noexcept;
  Workspace(const Workspace &) = delete;
  Workspace &operator=(const Workspace &) = delete;

  // Allocates device and pinned-host storage and uploads the topology plan.
  // Repeated prepared calls do not allocate memory or rebuild the plan.
  void Reserve(const btrc::Plan &plan, std::size_t states, std::size_t batch,
               int device = 0);

private:
  friend tree_hmm::PartitionView
  PartitionFunctionPrepared(tree_hmm::BatchedModelView, Workspace &);
  friend tree_hmm::PartitionView
  LogPartitionFunctionPrepared(tree_hmm::BatchedModelView, Workspace &);
  std::unique_ptr<Impl> impl_;
};

tree_hmm::PartitionView
PartitionFunctionPrepared(tree_hmm::BatchedModelView model,
                          Workspace &workspace);
tree_hmm::PartitionView
LogPartitionFunctionPrepared(tree_hmm::BatchedModelView model,
                             Workspace &workspace);

} // namespace tree_hmm::cuda

#endif // TREE_HMM_CUDA_H_
