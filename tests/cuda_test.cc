#include "tree_hmm/cuda.h"

#include "accelerator_test.h"

int main() {
  tree_hmm::cuda::Workspace workspace;
  TestAccelerator(
      "CUDA", tree_hmm::cuda::Available(),
      [&](const btrc::Plan &plan, std::size_t states, std::size_t batch) {
        workspace.Reserve(plan, states, batch);
      },
      [&] { return workspace.Inputs(); },
      [&](tree_hmm::BatchedModelView model) {
        return tree_hmm::cuda::PartitionFunctionPrepared(model, workspace);
      },
      [&](tree_hmm::BatchedModelView model) {
        return tree_hmm::cuda::LogPartitionFunctionPrepared(model, workspace);
      });
}
