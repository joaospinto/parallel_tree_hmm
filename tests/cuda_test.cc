#include "tree_hmm/cuda.h"

#include "accelerator_test.h"

int main() {
  tree_hmm::cuda::Workspace workspace;
  TestAccelerator(
      "CUDA", tree_hmm::cuda::Available(),
      [&](const btrc::Plan &plan, std::size_t states, std::size_t batch) {
        workspace.Reserve(plan, states, batch);
      },
      [&](std::size_t batch) { return workspace.Inputs(batch); },
      [&](tree_hmm::BatchedModelView model) {
        return tree_hmm::cuda::PartitionFunctionPrepared(model, workspace);
      },
      [&](tree_hmm::BatchedModelView model) {
        return tree_hmm::cuda::LogPartitionFunctionPrepared(model, workspace);
      });
  TestCategoricalAccelerator(
      "CUDA", tree_hmm::cuda::Available(),
      [&](const btrc::Plan &plan, std::size_t states, std::size_t batch,
          std::size_t categories,
          std::span<const btrc::Index> observation_nodes) {
        workspace.ReserveCategorical(plan, states, batch, categories,
                                     observation_nodes);
      },
      [&](std::size_t batch) { return workspace.CategoricalInputs(batch); },
      [&](tree_hmm::BatchedCategoricalModelView model) {
        return tree_hmm::cuda::LogPartitionFunctionPrepared(model, workspace);
      });
}
