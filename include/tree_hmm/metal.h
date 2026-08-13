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

  // Allocates numerical buffers for up to batch_capacity items, copies the
  // topology plan, and compiles no topology-specific kernels. Repeated
  // prepared calls allocate no numerical workspace storage.
  void Reserve(const btrc::Plan &plan, std::size_t states,
               std::size_t batch_capacity);
  void ReserveCategorical(const btrc::Plan &plan, std::size_t states,
                          std::size_t batch_capacity, std::size_t categories,
                          std::span<const btrc::Index> observation_nodes);
  void ReserveMaximum(const btrc::Plan &plan, std::size_t states,
                      std::size_t batch_capacity);
  void
  ReserveCategoricalMaximum(const btrc::Plan &plan, std::size_t states,
                            std::size_t batch_capacity, std::size_t categories,
                            std::span<const btrc::Index> observation_nodes);
  void ReserveSampling(const btrc::Plan &plan, std::size_t states,
                       std::size_t batch_capacity);
  void
  ReserveCategoricalSampling(const btrc::Plan &plan, std::size_t states,
                             std::size_t batch_capacity, std::size_t categories,
                             std::span<const btrc::Index> observation_nodes);
  void ReserveMarginals(const btrc::Plan &plan, std::size_t states,
                        std::size_t batch_capacity);
  void ReserveCategoricalMarginals(
      const btrc::Plan &plan, std::size_t states, std::size_t batch_capacity,
      std::size_t categories, std::span<const btrc::Index> observation_nodes);

  // Returns shared host/device storage owned by this workspace. Preparing
  // factors directly into it eliminates an otherwise redundant copy.
  tree_hmm::MutableBatchedModelView Inputs();
  tree_hmm::MutableBatchedModelView Inputs(std::size_t batch);
  tree_hmm::MutableBatchedCategoricalModelView CategoricalInputs();
  tree_hmm::MutableBatchedCategoricalModelView
  CategoricalInputs(std::size_t batch);
  std::span<Scalar> Uniforms();
  std::span<Scalar> Uniforms(std::size_t batch);

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
  friend tree_hmm::BatchedPosteriorSampleView
  PosteriorSamplePrepared(tree_hmm::BatchedModelView, std::span<const Scalar>,
                          Workspace &);
  friend tree_hmm::BatchedPosteriorSampleView
  PosteriorSamplePrepared(tree_hmm::BatchedCategoricalModelView,
                          std::span<const Scalar>, Workspace &);
  friend tree_hmm::BatchedMarginalView
  PosteriorMarginalsPrepared(tree_hmm::BatchedModelView, Workspace &);
  friend tree_hmm::BatchedMarginalView
  PosteriorMarginalsPrepared(tree_hmm::BatchedCategoricalModelView,
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
tree_hmm::BatchedPosteriorSampleView
PosteriorSamplePrepared(tree_hmm::BatchedModelView model,
                        std::span<const Scalar> uniforms, Workspace &workspace);
tree_hmm::BatchedPosteriorSampleView
PosteriorSamplePrepared(tree_hmm::BatchedCategoricalModelView model,
                        std::span<const Scalar> uniforms, Workspace &workspace);
tree_hmm::BatchedMarginalView
PosteriorMarginalsPrepared(tree_hmm::BatchedModelView model,
                           Workspace &workspace);
tree_hmm::BatchedMarginalView
PosteriorMarginalsPrepared(tree_hmm::BatchedCategoricalModelView model,
                           Workspace &workspace);

} // namespace tree_hmm::metal

#endif // TREE_HMM_METAL_H_
