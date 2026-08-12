#ifndef TREE_HMM_METAL_H_
#define TREE_HMM_METAL_H_

#include <cstddef>
#include <memory>
#include <string>
#include "tree_hmm/accelerator.h"

namespace tree_hmm::metal {

bool Available();
std::string DeviceDescription();

class Workspace {
public:
  struct Impl;

  Workspace();
  ~Workspace();
  Workspace(Workspace &&) noexcept;
  Workspace &operator=(Workspace &&) noexcept;
  Workspace(const Workspace &) = delete;
  Workspace &operator=(const Workspace &) = delete;

  // Allocates buffers, copies the topology plan, and compiles no
  // topology-specific kernels. Repeated prepared calls allocate nothing.
  void Reserve(const btrc::Plan &plan, std::size_t states, std::size_t batch);

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

} // namespace tree_hmm::metal

#endif // TREE_HMM_METAL_H_
