int main() {
  tree_hmm::TREE_HMM_GPU_TEST_NAMESPACE::Workspace workspace;
  tree_hmm::TREE_HMM_GPU_TEST_NAMESPACE::Workspace dense_workspace;
  TestAccelerator(
      TREE_HMM_GPU_TEST_NAME, tree_hmm::TREE_HMM_GPU_TEST_NAMESPACE::Available(),
      [&](const btrc::Plan &plan, std::size_t states, std::size_t batch) {
        workspace.Reserve(plan, states, batch);
      },
      [&](std::size_t batch) { return workspace.Inputs(batch); },
      [&](tree_hmm::BatchedModelView model) {
        return tree_hmm::TREE_HMM_GPU_TEST_NAMESPACE::PartitionFunctionPrepared(model, workspace);
      },
      [&](tree_hmm::BatchedModelView model) {
        return tree_hmm::TREE_HMM_GPU_TEST_NAMESPACE::LogPartitionFunctionPrepared(model, workspace);
      });
  TestCategoricalAccelerator(
      TREE_HMM_GPU_TEST_NAME, tree_hmm::TREE_HMM_GPU_TEST_NAMESPACE::Available(),
      [&](const btrc::Plan &plan, std::size_t states, std::size_t batch,
          std::size_t categories,
          std::span<const btrc::Index> observation_nodes) {
        workspace.ReserveCategoricalMaximum(plan, states, batch, categories,
                                            observation_nodes);
      },
      [&](std::size_t batch) { return workspace.CategoricalInputs(batch); },
      [&](tree_hmm::BatchedCategoricalModelView model) {
        return tree_hmm::TREE_HMM_GPU_TEST_NAMESPACE::LogPartitionFunctionPrepared(model, workspace);
      },
      [&](tree_hmm::BatchedCategoricalModelView model) {
        return tree_hmm::TREE_HMM_GPU_TEST_NAMESPACE::MaximumAPosterioriPrepared(model, workspace);
      },
      [&](tree_hmm::BatchedModelView model) {
        dense_workspace.Reserve(model.plan, model.states, model.batch);
        return tree_hmm::TREE_HMM_GPU_TEST_NAMESPACE::LogPartitionFunctionPrepared(model,
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
        return tree_hmm::TREE_HMM_GPU_TEST_NAMESPACE::PosteriorSamplePrepared(model, uniforms,
                                                       workspace);
      },
      [&](const btrc::Plan &plan, std::size_t states, std::size_t batch,
          std::size_t categories,
          std::span<const btrc::Index> observation_nodes) {
        workspace.ReserveCategoricalMarginals(plan, states, batch, categories,
                                              observation_nodes);
      },
      [&](tree_hmm::BatchedCategoricalModelView model) {
        return tree_hmm::TREE_HMM_GPU_TEST_NAMESPACE::PosteriorMarginalsPrepared(model, workspace);
      });
  TestMaximumAccelerator(
      TREE_HMM_GPU_TEST_NAME, tree_hmm::TREE_HMM_GPU_TEST_NAMESPACE::Available(),
      [&](const btrc::Plan &plan, std::size_t states, std::size_t batch) {
        workspace.ReserveMaximum(plan, states, batch);
      },
      [&](std::size_t batch) { return workspace.Inputs(batch); },
      [&](tree_hmm::BatchedModelView model) {
        return tree_hmm::TREE_HMM_GPU_TEST_NAMESPACE::MaximumAPosterioriPrepared(model, workspace);
      });
  TestSamplingAccelerator(
      TREE_HMM_GPU_TEST_NAME, tree_hmm::TREE_HMM_GPU_TEST_NAMESPACE::Available(),
      [&](const btrc::Plan &plan, std::size_t states, std::size_t batch) {
        workspace.ReserveSampling(plan, states, batch);
      },
      [&](std::size_t batch) { return workspace.Inputs(batch); },
      [&](std::size_t batch) { return workspace.Uniforms(batch); },
      [&](tree_hmm::BatchedModelView model,
          std::span<const tree_hmm::Scalar> uniforms) {
        return tree_hmm::TREE_HMM_GPU_TEST_NAMESPACE::PosteriorSamplePrepared(model, uniforms,
                                                       workspace);
      });
  TestMarginalAccelerator(
      TREE_HMM_GPU_TEST_NAME, tree_hmm::TREE_HMM_GPU_TEST_NAMESPACE::Available(),
      [&](const btrc::Plan &plan, std::size_t states, std::size_t batch) {
        workspace.ReserveMarginals(plan, states, batch);
      },
      [&](std::size_t batch) { return workspace.Inputs(batch); },
      [&](tree_hmm::BatchedModelView model) {
        return tree_hmm::TREE_HMM_GPU_TEST_NAMESPACE::PosteriorMarginalsPrepared(model, workspace);
      });
}
