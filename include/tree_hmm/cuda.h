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

  // Allocates device and pinned-host storage for up to batch_capacity items
  // and uploads the topology plan. Repeated prepared calls do not allocate
  // memory or rebuild the plan.
  void Reserve(const btrc::Plan &plan, std::size_t states,
               std::size_t batch_capacity, int device = 0);

  // Reserves the same numerical executor with compact categorical node
  // observations. Observation locations are topology data and are uploaded
  // only when the workspace is reserved.
  void ReserveCategorical(const btrc::Plan &plan, std::size_t states,
                          std::size_t batch_capacity, std::size_t categories,
                          std::span<const btrc::Index> observation_nodes,
                          int device = 0);

  // Reserves the forward workspace together with reconstruction storage.
  // Likelihood-only Reserve calls remain smaller and use the same kernels as
  // before. This workspace can evaluate maximum-a-posteriori assignments.
  void ReserveBidirectional(const btrc::Plan &plan, std::size_t states,
                            std::size_t batch_capacity, int device = 0);
  void ReserveCategoricalBidirectional(
      const btrc::Plan &plan, std::size_t states, std::size_t batch_capacity,
      std::size_t categories, std::span<const btrc::Index> observation_nodes,
      int device = 0);

  // Returns pinned host storage owned by this workspace. Preparing factors
  // directly into it eliminates an otherwise redundant host-side copy.
  tree_hmm::MutableBatchedModelView Inputs();
  tree_hmm::MutableBatchedModelView Inputs(std::size_t batch);
  tree_hmm::MutableBatchedCategoricalModelView CategoricalInputs();
  tree_hmm::MutableBatchedCategoricalModelView
  CategoricalInputs(std::size_t batch);

private:
  friend tree_hmm::PartitionView
  PartitionFunctionPrepared(tree_hmm::BatchedModelView, Workspace &);
  friend tree_hmm::PartitionView
  LogPartitionFunctionPrepared(tree_hmm::BatchedModelView, Workspace &);
  friend tree_hmm::PartitionView
  PartitionFunctionPrepared(tree_hmm::BatchedCategoricalModelView, Workspace &);
  friend tree_hmm::PartitionView
  LogPartitionFunctionPrepared(tree_hmm::BatchedCategoricalModelView,
                               Workspace &);
  friend tree_hmm::BatchedMaximumAssignmentView
  MaximumAPosterioriPrepared(tree_hmm::BatchedModelView, Workspace &);
  friend tree_hmm::BatchedMaximumAssignmentView
  MaximumAPosterioriPrepared(tree_hmm::BatchedCategoricalModelView,
                             Workspace &);
  std::unique_ptr<Impl> impl_;
};

tree_hmm::PartitionView
PartitionFunctionPrepared(tree_hmm::BatchedModelView model,
                          Workspace &workspace);
tree_hmm::PartitionView
LogPartitionFunctionPrepared(tree_hmm::BatchedModelView model,
                             Workspace &workspace);
tree_hmm::PartitionView
PartitionFunctionPrepared(tree_hmm::BatchedCategoricalModelView model,
                          Workspace &workspace);
tree_hmm::PartitionView
LogPartitionFunctionPrepared(tree_hmm::BatchedCategoricalModelView model,
                             Workspace &workspace);
tree_hmm::BatchedMaximumAssignmentView
MaximumAPosterioriPrepared(tree_hmm::BatchedModelView model,
                           Workspace &workspace);
tree_hmm::BatchedMaximumAssignmentView
MaximumAPosterioriPrepared(tree_hmm::BatchedCategoricalModelView model,
                           Workspace &workspace);

} // namespace tree_hmm::cuda

#endif // TREE_HMM_CUDA_H_
