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
  TestMaximumAccelerator(
      "Metal", tree_hmm::metal::Available(),
      [&](const btrc::Plan &plan, std::size_t states, std::size_t batch) {
        workspace.ReserveMaximum(plan, states, batch);
      },
      [&](std::size_t batch) { return workspace.Inputs(batch); },
      [&](tree_hmm::BatchedModelView model) {
        return tree_hmm::metal::MaximumAPosterioriPrepared(model, workspace);
      });
  TestSamplingAccelerator(
      "Metal", tree_hmm::metal::Available(),
      [&](const btrc::Plan &plan, std::size_t states, std::size_t batch) {
        workspace.ReserveSampling(plan, states, batch);
      },
      [&](std::size_t batch) { return workspace.Inputs(batch); },
      [&](std::size_t batch) { return workspace.Uniforms(batch); },
      [&](tree_hmm::BatchedModelView model, std::span<const float> uniforms) {
        return tree_hmm::metal::PosteriorSamplePrepared(model, uniforms,
                                                        workspace);
      });
}
