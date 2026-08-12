#include "tree_hmm/metal.h"

#include "accelerator_test.h"

int main() {
  tree_hmm::metal::Workspace workspace;
  TestAccelerator(
      "Metal", tree_hmm::metal::Available(),
      [&](const btrc::Plan &plan, std::size_t states, std::size_t batch) {
        workspace.Reserve(plan, states, batch);
      },
      [&](std::size_t batch) { return workspace.Inputs(batch); },
      [&](tree_hmm::BatchedModelView model) {
        return tree_hmm::metal::PartitionFunctionPrepared(model, workspace);
      },
      [&](tree_hmm::BatchedModelView model) {
        return tree_hmm::metal::LogPartitionFunctionPrepared(model, workspace);
      });
}
