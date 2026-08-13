#include "tree_hmm/cuda.h"

#include "accelerator_test.h"

int main() {
  tree_hmm::cuda::Workspace workspace;
  tree_hmm::cuda::Workspace dense_workspace;
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
        workspace.ReserveCategoricalMaximum(plan, states, batch, categories,
                                            observation_nodes);
      },
      [&](std::size_t batch) { return workspace.CategoricalInputs(batch); },
      [&](tree_hmm::BatchedCategoricalModelView model) {
        return tree_hmm::cuda::LogPartitionFunctionPrepared(model, workspace);
      },
      [&](tree_hmm::BatchedCategoricalModelView model) {
        return tree_hmm::cuda::MaximumAPosterioriPrepared(model, workspace);
      },
      [&](tree_hmm::BatchedModelView model) {
        dense_workspace.Reserve(model.plan, model.states, model.batch);
        return tree_hmm::cuda::LogPartitionFunctionPrepared(model,
                                                            dense_workspace);
      },
      [&](const btrc::Plan &plan, std::size_t states, std::size_t batch,
          std::size_t categories,
          std::span<const btrc::Index> observation_nodes) {
        workspace.ReserveCategoricalSampling(plan, states, batch, categories,
                                             observation_nodes);
      },
      [&](std::size_t batch) { return workspace.Uniforms(batch); },
      [&](tree_hmm::BatchedCategoricalModelView model,
          std::span<const tree_hmm::Scalar> uniforms) {
        return tree_hmm::cuda::PosteriorSamplePrepared(model, uniforms,
                                                       workspace);
      },
      [&](const btrc::Plan &plan, std::size_t states, std::size_t batch,
          std::size_t categories,
          std::span<const btrc::Index> observation_nodes) {
        workspace.ReserveCategoricalMarginals(plan, states, batch, categories,
                                              observation_nodes);
      },
      [&](tree_hmm::BatchedCategoricalModelView model) {
        return tree_hmm::cuda::PosteriorMarginalsPrepared(model, workspace);
      });
  TestMaximumAccelerator(
      "CUDA", tree_hmm::cuda::Available(),
      [&](const btrc::Plan &plan, std::size_t states, std::size_t batch) {
        workspace.ReserveMaximum(plan, states, batch);
      },
      [&](std::size_t batch) { return workspace.Inputs(batch); },
      [&](tree_hmm::BatchedModelView model) {
        return tree_hmm::cuda::MaximumAPosterioriPrepared(model, workspace);
      });
  TestSamplingAccelerator(
      "CUDA", tree_hmm::cuda::Available(),
      [&](const btrc::Plan &plan, std::size_t states, std::size_t batch) {
        workspace.ReserveSampling(plan, states, batch);
      },
      [&](std::size_t batch) { return workspace.Inputs(batch); },
      [&](std::size_t batch) { return workspace.Uniforms(batch); },
      [&](tree_hmm::BatchedModelView model,
          std::span<const tree_hmm::Scalar> uniforms) {
        return tree_hmm::cuda::PosteriorSamplePrepared(model, uniforms,
                                                       workspace);
      });
  TestMarginalAccelerator(
      "CUDA", tree_hmm::cuda::Available(),
      [&](const btrc::Plan &plan, std::size_t states, std::size_t batch) {
        workspace.ReserveMarginals(plan, states, batch);
      },
      [&](std::size_t batch) { return workspace.Inputs(batch); },
      [&](tree_hmm::BatchedModelView model) {
        return tree_hmm::cuda::PosteriorMarginalsPrepared(model, workspace);
      });
}
