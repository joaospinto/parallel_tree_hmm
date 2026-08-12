#ifndef TREE_HMM_METAL_H_
#define TREE_HMM_METAL_H_

#include "tree_hmm/accelerator.h"
#include <cstddef>
#include <memory>
#include <string>

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

  // Allocates buffers for up to batch_capacity items, copies the topology
  // plan, and compiles no topology-specific kernels. Repeated prepared calls
  // allocate nothing.
  void Reserve(const btrc::Plan &plan, std::size_t states,
               std::size_t batch_capacity);
  void ReserveBidirectional(const btrc::Plan &plan, std::size_t states,
                            std::size_t batch_capacity);

  // Returns shared host/device storage owned by this workspace. Preparing
  // factors directly into it eliminates an otherwise redundant copy.
  tree_hmm::MutableBatchedModelView Inputs();
  tree_hmm::MutableBatchedModelView Inputs(std::size_t batch);

private:
  friend tree_hmm::PartitionView
  PartitionFunctionPrepared(tree_hmm::BatchedModelView, Workspace &);
  friend tree_hmm::PartitionView
  LogPartitionFunctionPrepared(tree_hmm::BatchedModelView, Workspace &);
  friend tree_hmm::BatchedMaximumAssignmentView
  MaximumAPosterioriPrepared(tree_hmm::BatchedModelView, Workspace &);
  std::unique_ptr<Impl> impl_;
};

tree_hmm::PartitionView
PartitionFunctionPrepared(tree_hmm::BatchedModelView model,
                          Workspace &workspace);
tree_hmm::PartitionView
LogPartitionFunctionPrepared(tree_hmm::BatchedModelView model,
                             Workspace &workspace);
tree_hmm::BatchedMaximumAssignmentView
MaximumAPosterioriPrepared(tree_hmm::BatchedModelView model,
                           Workspace &workspace);

} // namespace tree_hmm::metal

#endif // TREE_HMM_METAL_H_
