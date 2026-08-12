#ifndef TREE_HMM_TESTS_ACCELERATOR_TEST_H_
#define TREE_HMM_TESTS_ACCELERATOR_TEST_H_

#include "tree_hmm/inference.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

template <class Reserve, class Inputs, class Solve, class SolveLog>
void TestAccelerator(const char *name, bool available, Reserve reserve,
                     Inputs inputs, Solve solve, SolveLog solve_log) {
  if (!available)
    return;
  const btrc::Plan plan =
      btrc::MakePlan(std::vector<std::int64_t>{-1, 0, 0, 1, 1, 2, 2});
  constexpr std::size_t kStates = 4;
  constexpr std::size_t kBatch = 7;
  std::vector<float> nodes(kBatch * plan.num_nodes() * kStates);
  for (std::size_t batch = 0; batch < kBatch; ++batch) {
    for (std::size_t node = 0; node < plan.num_nodes(); ++node) {
      for (std::size_t state = 0; state < kStates; ++state) {
        nodes[(batch * plan.num_nodes() + node) * kStates + state] =
            0.2f + 0.01f * static_cast<float>(1 + batch + 2 * node + state);
      }
    }
  }
  std::vector<float> edges(plan.num_edges() * kStates * kStates);
  for (std::size_t edge = 0; edge < plan.num_edges(); ++edge) {
    for (std::size_t parent = 0; parent < kStates; ++parent) {
      for (std::size_t child = 0; child < kStates; ++child) {
        edges[(edge * kStates + parent) * kStates + child] =
            parent == child ? 0.82f : 0.06f;
      }
    }
  }

  reserve(plan, kStates, kBatch);
  const tree_hmm::PartitionView result =
      solve(tree_hmm::BatchedModelView{plan, kStates, kBatch, nodes, edges});
  if (result.values.size() != kBatch)
    throw std::runtime_error(std::string(name) + " output shape is wrong");
  for (std::size_t batch = 0; batch < kBatch; ++batch) {
    std::vector<double> host_nodes(plan.num_nodes() * kStates);
    std::transform(nodes.begin() + batch * host_nodes.size(),
                   nodes.begin() + (batch + 1) * host_nodes.size(),
                   host_nodes.begin(),
                   [](float value) { return static_cast<double>(value); });
    std::vector<double> host_edges(edges.begin(), edges.end());
    const double expected =
        tree_hmm::PartitionFunction({plan, kStates, host_nodes, host_edges});
    const double actual = result.values[batch];
    const double tolerance =
        2e-5 * std::max({1.0, std::abs(actual), std::abs(expected)});
    if (std::abs(actual - expected) > tolerance) {
      throw std::runtime_error(std::string(name) +
                               " disagrees with CPU inference");
    }
  }
  if (result.timings.kernel_ms < 0.0)
    throw std::runtime_error(std::string(name) + " kernel timing is invalid");

  tree_hmm::MutableBatchedModelView staged = inputs(kBatch);
  std::copy(nodes.begin(), nodes.end(), staged.node_potentials.begin());
  std::copy(edges.begin(), edges.end(), staged.edge_potentials.begin());
  const tree_hmm::PartitionView staged_result =
      solve(static_cast<tree_hmm::BatchedModelView>(staged));
  for (std::size_t batch = 0; batch < kBatch; ++batch) {
    std::vector<double> host_nodes(plan.num_nodes() * kStates);
    std::transform(nodes.begin() + batch * host_nodes.size(),
                   nodes.begin() + (batch + 1) * host_nodes.size(),
                   host_nodes.begin(),
                   [](float value) { return static_cast<double>(value); });
    const std::vector<double> host_edges(edges.begin(), edges.end());
    const double expected =
        tree_hmm::PartitionFunction({plan, kStates, host_nodes, host_edges});
    if (std::abs(staged_result.values[batch] - expected) >
        2e-5 * std::max(1.0, std::abs(expected))) {
      throw std::runtime_error(std::string(name) +
                               " staged input disagrees with CPU inference");
    }
  }

  constexpr std::size_t kTailBatch = 3;
  tree_hmm::MutableBatchedModelView tail = inputs(kTailBatch);
  std::copy(nodes.begin(),
            nodes.begin() + kTailBatch * plan.num_nodes() * kStates,
            tail.node_potentials.begin());
  std::copy(edges.begin(), edges.end(), tail.edge_potentials.begin());
  const tree_hmm::PartitionView tail_result =
      solve_log(static_cast<tree_hmm::BatchedModelView>(tail));
  if (tail_result.values.size() != kTailBatch)
    throw std::runtime_error(std::string(name) +
                             " capacity-backed output shape is wrong");
  for (std::size_t batch = 0; batch < kTailBatch; ++batch) {
    const std::size_t node_values = plan.num_nodes() * kStates;
    const std::vector<double> host_nodes(nodes.begin() + batch * node_values,
                                         nodes.begin() +
                                             (batch + 1) * node_values);
    const std::vector<double> host_edges(edges.begin(), edges.end());
    const double expected =
        tree_hmm::LogPartitionFunction({plan, kStates, host_nodes, host_edges});
    if (std::abs(tail_result.values[batch] - expected) >
        5e-5 * std::max(1.0, std::abs(expected))) {
      throw std::runtime_error(std::string(name) +
                               " capacity-backed inference disagrees with "
                               "CPU");
    }
  }

  const tree_hmm::PartitionView log_result = solve_log(
      tree_hmm::BatchedModelView{plan, kStates, kBatch, nodes, edges});
  for (std::size_t batch = 0; batch < kBatch; ++batch) {
    std::vector<double> host_nodes(plan.num_nodes() * kStates);
    std::transform(nodes.begin() + batch * host_nodes.size(),
                   nodes.begin() + (batch + 1) * host_nodes.size(),
                   host_nodes.begin(),
                   [](float value) { return static_cast<double>(value); });
    std::vector<double> host_edges(edges.begin(), edges.end());
    const double expected =
        tree_hmm::LogPartitionFunction({plan, kStates, host_nodes, host_edges});
    const double actual = log_result.values[batch];
    const double tolerance =
        5e-5 * std::max({1.0, std::abs(actual), std::abs(expected)});
    if (std::abs(actual - expected) > tolerance) {
      throw std::runtime_error(std::string(name) +
                               " scaled inference disagrees with CPU");
    }
  }

  // A unary chain forces path composition and therefore site-specific path
  // storage. Its edge factors vary by edge so accidental path broadcasting is
  // observable.
  const btrc::Plan chain_plan =
      btrc::MakePlan(std::vector<std::int64_t>{-1, 0, 1, 2, 3, 4, 5});
  constexpr std::size_t kChainBatch = 5;
  std::vector<float> chain_nodes(kChainBatch * chain_plan.num_nodes() *
                                 kStates);
  for (std::size_t index = 0; index < chain_nodes.size(); ++index)
    chain_nodes[index] = 0.15f + 0.007f * static_cast<float>(index % 17);
  std::vector<float> chain_edges(chain_plan.num_edges() * kStates * kStates);
  for (std::size_t edge = 0; edge < chain_plan.num_edges(); ++edge) {
    for (std::size_t parent = 0; parent < kStates; ++parent) {
      for (std::size_t child = 0; child < kStates; ++child) {
        chain_edges[(edge * kStates + parent) * kStates + child] =
            0.02f +
            0.01f * static_cast<float>(1 + edge + 2 * parent + 3 * child);
      }
    }
  }
  reserve(chain_plan, kStates, kChainBatch);
  const tree_hmm::PartitionView chain_result =
      solve_log(tree_hmm::BatchedModelView{chain_plan, kStates, kChainBatch,
                                           chain_nodes, chain_edges});
  for (std::size_t batch = 0; batch < kChainBatch; ++batch) {
    const std::size_t node_values = chain_plan.num_nodes() * kStates;
    const std::vector<double> host_nodes(
        chain_nodes.begin() + batch * node_values,
        chain_nodes.begin() + (batch + 1) * node_values);
    const std::vector<double> host_edges(chain_edges.begin(),
                                         chain_edges.end());
    const double expected = tree_hmm::LogPartitionFunction(
        {chain_plan, kStates, host_nodes, host_edges});
    const double actual = chain_result.values[batch];
    if (std::abs(actual - expected) >
        5e-5 * std::max({1.0, std::abs(actual), std::abs(expected)})) {
      throw std::runtime_error(std::string(name) +
                               " path composition disagrees with CPU");
    }
  }

  constexpr std::size_t kLongNodes = 4096;
  std::vector<std::int64_t> long_parents(kLongNodes);
  long_parents[0] = -1;
  for (std::size_t node = 1; node < kLongNodes; ++node)
    long_parents[node] = static_cast<std::int64_t>(node - 1);
  const btrc::Plan long_plan = btrc::MakePlan(long_parents);
  const std::vector<float> long_nodes(kLongNodes, 0.5f);
  const std::vector<float> long_edges(kLongNodes - 1, 1.0f);
  reserve(long_plan, 1, 1);
  const tree_hmm::PartitionView long_result = solve_log(
      tree_hmm::BatchedModelView{long_plan, 1, 1, long_nodes, long_edges});
  const double long_expected = kLongNodes * std::log(0.5);
  if (std::abs(long_result.values[0] - long_expected) > 2e-3) {
    throw std::runtime_error(std::string(name) +
                             " scaled inference underflowed on a long tree");
  }

  // Exercise the generic-state fallback, including path composition. The
  // optimized small-state kernels above must not become the only tested path.
  constexpr std::size_t kGenericStates = 9;
  constexpr std::size_t kGenericBatch = 2;
  const btrc::Plan generic_plan =
      btrc::MakePlan(std::vector<std::int64_t>{-1, 0, 1, 2, 3});
  std::vector<float> generic_nodes(kGenericBatch * generic_plan.num_nodes() *
                                   kGenericStates);
  for (std::size_t index = 0; index < generic_nodes.size(); ++index)
    generic_nodes[index] = 0.1f + 0.003f * static_cast<float>(index % 23);
  std::vector<float> generic_edges(generic_plan.num_edges() * kGenericStates *
                                   kGenericStates);
  for (std::size_t index = 0; index < generic_edges.size(); ++index)
    generic_edges[index] = 0.02f + 0.001f * static_cast<float>(index % 31);
  reserve(generic_plan, kGenericStates, kGenericBatch);
  const tree_hmm::PartitionView generic_result =
      solve_log({generic_plan, kGenericStates, kGenericBatch, generic_nodes,
                 generic_edges});
  for (std::size_t batch = 0; batch < kGenericBatch; ++batch) {
    const std::size_t node_values = generic_plan.num_nodes() * kGenericStates;
    const std::vector<double> host_nodes(
        generic_nodes.begin() + batch * node_values,
        generic_nodes.begin() + (batch + 1) * node_values);
    const std::vector<double> host_edges(generic_edges.begin(),
                                         generic_edges.end());
    const double expected = tree_hmm::LogPartitionFunction(
        {generic_plan, kGenericStates, host_nodes, host_edges});
    const double actual = generic_result.values[batch];
    if (std::abs(actual - expected) >
        1e-4 * std::max({1.0, std::abs(actual), std::abs(expected)})) {
      throw std::runtime_error(std::string(name) +
                               " generic-state inference disagrees with CPU");
    }
  }
}

template <class Reserve, class Inputs, class SolveLog, class SolveMaximum,
          class SolveDense, class ReserveSampling, class Uniforms,
          class SolveSampling, class ReserveMarginals, class SolveMarginals>
void TestCategoricalAccelerator(const char *name, bool available,
                                Reserve reserve, Inputs inputs,
                                SolveLog solve_log, SolveMaximum solve_maximum,
                                SolveDense solve_dense,
                                ReserveSampling reserve_sampling,
                                Uniforms uniforms,
                                SolveSampling solve_sampling,
                                ReserveMarginals reserve_marginals,
                                SolveMarginals solve_marginals) {
  if (!available)
    return;
  const btrc::Plan plan =
      btrc::MakePlan(std::vector<std::int64_t>{-1, 0, 1, 1, 0, 4, 5, 5});
  constexpr std::size_t kStates = 4;
  constexpr std::size_t kBatch = 7;
  constexpr std::size_t kCategories = 6;
  const std::vector<btrc::Index> observation_nodes{2, 3, 6, 7};
  const std::vector<float> root_potential{0.1f, 0.2f, 0.3f, 0.4f};
  const std::vector<float> emissions{
      1.0f, 1.0f, 1.0f, 1.0f, // unobserved
      1.0f, 0.0f, 0.0f, 0.0f, // state 0
      0.0f, 1.0f, 0.0f, 0.0f, // state 1
      0.0f, 0.0f, 1.0f, 0.0f, // state 2
      0.0f, 0.0f, 0.0f, 1.0f, // state 3
      1.0f, 0.0f, 1.0f, 0.0f, // states 0 or 2
  };
  std::vector<std::uint8_t> observations(kBatch * observation_nodes.size());
  for (std::size_t batch = 0; batch < kBatch; ++batch) {
    for (std::size_t observation = 0; observation < observation_nodes.size();
         ++observation) {
      observations[batch * observation_nodes.size() + observation] =
          static_cast<std::uint8_t>((batch + 2 * observation) % kCategories);
    }
  }
  std::vector<float> edges(plan.num_edges() * kStates * kStates);
  for (std::size_t edge = 0; edge < plan.num_edges(); ++edge) {
    for (std::size_t parent = 0; parent < kStates; ++parent) {
      for (std::size_t child = 0; child < kStates; ++child) {
        edges[(edge * kStates + parent) * kStates + child] =
            parent == child ? 0.82f : 0.06f;
      }
    }
  }

  reserve(plan, kStates, kBatch, kCategories, observation_nodes);
  tree_hmm::MutableBatchedCategoricalModelView staged = inputs(kBatch);
  std::copy(observations.begin(), observations.end(),
            staged.observations.begin());
  std::copy(root_potential.begin(), root_potential.end(),
            staged.root_potential.begin());
  std::copy(emissions.begin(), emissions.end(),
            staged.emission_potentials.begin());
  std::copy(edges.begin(), edges.end(), staged.edge_potentials.begin());
  const tree_hmm::PartitionView result =
      solve_log(static_cast<tree_hmm::BatchedCategoricalModelView>(staged));
  const std::vector<float> log_values(result.values.begin(),
                                      result.values.end());
  const tree_hmm::BatchedMaximumAssignmentView maximum =
      solve_maximum(static_cast<tree_hmm::BatchedCategoricalModelView>(staged));
  std::vector<float> dense_nodes(kBatch * plan.num_nodes() * kStates, 1.0f);
  for (std::size_t batch = 0; batch < kBatch; ++batch) {
    std::copy(root_potential.begin(), root_potential.end(),
              dense_nodes.begin() + batch * plan.num_nodes() * kStates);
    for (std::size_t observation = 0; observation < observation_nodes.size();
         ++observation) {
      const std::uint8_t category =
          observations[batch * observation_nodes.size() + observation];
      for (std::size_t state = 0; state < kStates; ++state) {
        dense_nodes[(batch * plan.num_nodes() +
                     observation_nodes[observation]) *
                        kStates +
                    state] *= emissions[category * kStates + state];
      }
    }
  }
  const tree_hmm::PartitionView dense =
      solve_dense({plan, kStates, kBatch, dense_nodes, edges});
  for (std::size_t batch = 0; batch < kBatch; ++batch) {
    std::vector<double> nodes(plan.num_nodes() * kStates, 1.0);
    std::copy(root_potential.begin(), root_potential.end(), nodes.begin());
    for (std::size_t observation = 0; observation < observation_nodes.size();
         ++observation) {
      const std::uint8_t category =
          observations[batch * observation_nodes.size() + observation];
      for (std::size_t state = 0; state < kStates; ++state) {
        nodes[observation_nodes[observation] * kStates + state] *=
            emissions[category * kStates + state];
      }
    }
    const std::vector<double> host_edges(edges.begin(), edges.end());
    const double expected =
        tree_hmm::LogPartitionFunction({plan, kStates, nodes, host_edges});
    const double actual = log_values[batch];
    if (std::abs(actual - expected) >
        5e-5 * std::max({1.0, std::abs(actual), std::abs(expected)})) {
      throw std::runtime_error(std::string(name) +
                               " categorical inference disagrees with CPU");
    }
    if (dense.values[batch] != log_values[batch]) {
      throw std::runtime_error(std::string(name) +
                               " categorical and dense inference differ");
    }
    const tree_hmm::MaximumAssignment expected_maximum =
        tree_hmm::MaximumAPosteriori({plan, kStates, nodes, host_edges});
    if (std::abs(maximum.log_weights[batch] - expected_maximum.log_weight) >
        2e-5) {
      throw std::runtime_error(std::string(name) +
                               " categorical MAP weight disagrees with CPU");
    }
    for (std::size_t node = 0; node < plan.num_nodes(); ++node) {
      if (maximum.states[batch * plan.num_nodes() + node] !=
          expected_maximum.states[node]) {
        throw std::runtime_error(
            std::string(name) +
            " categorical MAP assignment disagrees with CPU");
      }
    }
  }

  constexpr std::size_t kTailBatch = 3;
  tree_hmm::MutableBatchedCategoricalModelView tail = inputs(kTailBatch);
  if (tail.observations.size() != kTailBatch * observation_nodes.size()) {
    throw std::runtime_error(std::string(name) +
                             " categorical tail shape is wrong");
  }

  reserve_sampling(plan, kStates, kBatch, kCategories, observation_nodes);
  tree_hmm::MutableBatchedCategoricalModelView sampling_staged =
      inputs(kBatch);
  std::copy(observations.begin(), observations.end(),
            sampling_staged.observations.begin());
  std::copy(root_potential.begin(), root_potential.end(),
            sampling_staged.root_potential.begin());
  std::copy(emissions.begin(), emissions.end(),
            sampling_staged.emission_potentials.begin());
  std::copy(edges.begin(), edges.end(),
            sampling_staged.edge_potentials.begin());
  std::span<float> staged_uniforms = uniforms(kBatch);
  for (std::size_t index = 0; index < staged_uniforms.size(); ++index)
    staged_uniforms[index] = 0.13f + 0.12f * static_cast<float>(index % 7);
  const tree_hmm::BatchedPosteriorSampleView samples =
      solve_sampling(
          static_cast<tree_hmm::BatchedCategoricalModelView>(sampling_staged),
          staged_uniforms);
  for (std::size_t batch = 0; batch < kBatch; ++batch) {
    std::vector<double> nodes(plan.num_nodes() * kStates, 1.0);
    std::copy(root_potential.begin(), root_potential.end(), nodes.begin());
    for (std::size_t observation = 0; observation < observation_nodes.size();
         ++observation) {
      const std::uint8_t category =
          observations[batch * observation_nodes.size() + observation];
      for (std::size_t state = 0; state < kStates; ++state) {
        nodes[observation_nodes[observation] * kStates + state] *=
            emissions[category * kStates + state];
      }
    }
    const std::vector<double> host_edges(edges.begin(), edges.end());
    const std::size_t offset = batch * plan.num_nodes();
    const std::vector<double> host_uniforms(staged_uniforms.begin() + offset,
                                            staged_uniforms.begin() + offset +
                                                plan.num_nodes());
    const std::vector<std::size_t> expected = tree_hmm::PosteriorSample(
        {plan, kStates, nodes, host_edges}, host_uniforms);
    for (std::size_t node = 0; node < plan.num_nodes(); ++node) {
      if (samples.states[batch * plan.num_nodes() + node] != expected[node]) {
        throw std::runtime_error(
            std::string(name) +
            " categorical posterior sample disagrees with CPU");
      }
    }
  }

  reserve_marginals(plan, kStates, kBatch, kCategories, observation_nodes);
  tree_hmm::MutableBatchedCategoricalModelView marginal_staged =
      inputs(kBatch);
  std::copy(observations.begin(), observations.end(),
            marginal_staged.observations.begin());
  std::copy(root_potential.begin(), root_potential.end(),
            marginal_staged.root_potential.begin());
  std::copy(emissions.begin(), emissions.end(),
            marginal_staged.emission_potentials.begin());
  std::copy(edges.begin(), edges.end(),
            marginal_staged.edge_potentials.begin());
  const tree_hmm::BatchedMarginalView marginals = solve_marginals(
      static_cast<tree_hmm::BatchedCategoricalModelView>(marginal_staged));
  const std::size_t nodes_per_batch = plan.num_nodes() * kStates;
  const std::size_t edges_per_batch = plan.num_edges() * kStates * kStates;
  if (marginals.log_partitions.size() != kBatch ||
      marginals.nodes.size() != kBatch * nodes_per_batch ||
      marginals.edges.size() != kBatch * edges_per_batch) {
    throw std::runtime_error(std::string(name) +
                             " categorical marginal shape is wrong");
  }
  for (std::size_t batch = 0; batch < kBatch; ++batch) {
    const std::vector<double> nodes(
        dense_nodes.begin() + batch * nodes_per_batch,
        dense_nodes.begin() + (batch + 1) * nodes_per_batch);
    const std::vector<double> host_edges(edges.begin(), edges.end());
    const tree_hmm::Marginals expected = tree_hmm::PosteriorMarginals(
        {plan, kStates, nodes, host_edges});
    if (std::abs(marginals.log_partitions[batch] -
                 expected.log_partition) >
        1e-4 * std::max(1.0, std::abs(expected.log_partition))) {
      throw std::runtime_error(
          std::string(name) +
          " categorical marginal log partition disagrees with CPU");
    }
    for (std::size_t index = 0; index < nodes_per_batch; ++index) {
      if (std::abs(marginals.nodes[batch * nodes_per_batch + index] -
                   expected.nodes[index]) > 8e-5) {
        throw std::runtime_error(
            std::string(name) +
            " categorical node marginal disagrees with CPU");
      }
    }
    for (std::size_t index = 0; index < edges_per_batch; ++index) {
      if (std::abs(marginals.edges[batch * edges_per_batch + index] -
                   expected.edges[index]) > 8e-5) {
        throw std::runtime_error(
            std::string(name) +
            " categorical edge marginal disagrees with CPU");
      }
    }
  }
}

template <class Reserve, class Inputs, class Solve>
void TestMaximumAccelerator(const char *name, bool available, Reserve reserve,
                            Inputs inputs, Solve solve) {
  if (!available)
    return;
  const btrc::Plan plan =
      btrc::MakePlan(std::vector<std::int64_t>{-1, 0, 1, 1, 0, 4, 5, 5});
  constexpr std::size_t kStates = 4;
  constexpr std::size_t kBatch = 5;
  std::vector<float> nodes(kBatch * plan.num_nodes() * kStates);
  for (std::size_t batch = 0; batch < kBatch; ++batch) {
    for (std::size_t node = 0; node < plan.num_nodes(); ++node) {
      for (std::size_t state = 0; state < kStates; ++state) {
        nodes[(batch * plan.num_nodes() + node) * kStates + state] =
            0.11f +
            0.013f * static_cast<float>(1 + 3 * batch + 5 * node + state);
      }
    }
  }
  std::vector<float> edges(plan.num_edges() * kStates * kStates);
  for (std::size_t edge = 0; edge < plan.num_edges(); ++edge) {
    for (std::size_t parent = 0; parent < kStates; ++parent) {
      for (std::size_t child = 0; child < kStates; ++child) {
        edges[(edge * kStates + parent) * kStates + child] =
            0.03f +
            0.007f * static_cast<float>(1 + edge + 2 * parent + 3 * child);
      }
    }
  }

  reserve(plan, kStates, kBatch);
  tree_hmm::MutableBatchedModelView staged = inputs(kBatch);
  std::copy(nodes.begin(), nodes.end(), staged.node_potentials.begin());
  std::copy(edges.begin(), edges.end(), staged.edge_potentials.begin());
  const tree_hmm::BatchedMaximumAssignmentView actual =
      solve(static_cast<tree_hmm::BatchedModelView>(staged));
  if (actual.log_weights.size() != kBatch ||
      actual.states.size() != kBatch * plan.num_nodes()) {
    throw std::runtime_error(std::string(name) + " MAP output shape is wrong");
  }
  for (std::size_t batch = 0; batch < kBatch; ++batch) {
    const std::size_t node_values = plan.num_nodes() * kStates;
    const std::vector<double> host_nodes(nodes.begin() + batch * node_values,
                                         nodes.begin() +
                                             (batch + 1) * node_values);
    const std::vector<double> host_edges(edges.begin(), edges.end());
    const tree_hmm::MaximumAssignment expected =
        tree_hmm::MaximumAPosteriori({plan, kStates, host_nodes, host_edges});
    if (std::abs(actual.log_weights[batch] - expected.log_weight) > 2e-5) {
      throw std::runtime_error(std::string(name) +
                               " MAP weight disagrees with CPU inference");
    }
    for (std::size_t node = 0; node < plan.num_nodes(); ++node) {
      if (actual.states[batch * plan.num_nodes() + node] !=
          expected.states[node]) {
        throw std::runtime_error(
            std::string(name) + " MAP assignment disagrees with CPU inference");
      }
    }
  }
}

template <class Reserve, class Inputs, class Uniforms, class Solve>
void TestSamplingAccelerator(const char *name, bool available, Reserve reserve,
                             Inputs inputs, Uniforms uniforms, Solve solve) {
  if (!available)
    return;
  const btrc::Plan plan =
      btrc::MakePlan(std::vector<std::int64_t>{-1, 0, 1, 1, 0, 4, 5, 5});
  constexpr std::size_t kStates = 4;
  constexpr std::size_t kBatch = 5;
  std::vector<float> nodes(kBatch * plan.num_nodes() * kStates);
  for (std::size_t batch = 0; batch < kBatch; ++batch) {
    for (std::size_t node = 0; node < plan.num_nodes(); ++node) {
      for (std::size_t state = 0; state < kStates; ++state) {
        nodes[(batch * plan.num_nodes() + node) * kStates + state] =
            0.2f +
            0.01f * static_cast<float>(1 + 2 * batch + 3 * node + 5 * state);
      }
    }
  }
  std::vector<float> edges(plan.num_edges() * kStates * kStates);
  for (std::size_t edge = 0; edge < plan.num_edges(); ++edge) {
    for (std::size_t parent = 0; parent < kStates; ++parent) {
      for (std::size_t child = 0; child < kStates; ++child) {
        edges[(edge * kStates + parent) * kStates + child] =
            0.1f +
            0.015f * static_cast<float>(1 + edge + 2 * parent + 4 * child);
      }
    }
  }
  std::vector<float> variates(kBatch * plan.num_nodes());
  for (std::size_t index = 0; index < variates.size(); ++index)
    variates[index] = 0.07f + 0.11f * static_cast<float>(index % 8);

  reserve(plan, kStates, kBatch);
  tree_hmm::MutableBatchedModelView staged = inputs(kBatch);
  std::copy(nodes.begin(), nodes.end(), staged.node_potentials.begin());
  std::copy(edges.begin(), edges.end(), staged.edge_potentials.begin());
  std::span<float> staged_uniforms = uniforms(kBatch);
  std::copy(variates.begin(), variates.end(), staged_uniforms.begin());
  const tree_hmm::BatchedPosteriorSampleView actual =
      solve(static_cast<tree_hmm::BatchedModelView>(staged), staged_uniforms);
  if (actual.states.size() != kBatch * plan.num_nodes()) {
    throw std::runtime_error(std::string(name) +
                             " posterior-sample shape is wrong");
  }
  for (std::size_t batch = 0; batch < kBatch; ++batch) {
    const std::size_t node_values = plan.num_nodes() * kStates;
    const std::vector<double> host_nodes(nodes.begin() + batch * node_values,
                                         nodes.begin() +
                                             (batch + 1) * node_values);
    const std::vector<double> host_edges(edges.begin(), edges.end());
    const std::size_t uniform_offset = batch * plan.num_nodes();
    const std::vector<double> host_uniforms(variates.begin() + uniform_offset,
                                            variates.begin() + uniform_offset +
                                                plan.num_nodes());
    const std::vector<std::size_t> expected = tree_hmm::PosteriorSample(
        {plan, kStates, host_nodes, host_edges}, host_uniforms);
    for (std::size_t node = 0; node < plan.num_nodes(); ++node) {
      if (actual.states[batch * plan.num_nodes() + node] != expected[node]) {
        throw std::runtime_error(std::string(name) +
                                 " posterior sample disagrees with CPU");
      }
    }
  }
  staged_uniforms[0] = 1.0f;
  bool invalid_uniform_rejected = false;
  try {
    static_cast<void>(solve(static_cast<tree_hmm::BatchedModelView>(staged),
                            staged_uniforms));
  } catch (const std::invalid_argument &) {
    invalid_uniform_rejected = true;
  }
  if (!invalid_uniform_rejected) {
    throw std::runtime_error(std::string(name) +
                             " accepted an invalid posterior uniform");
  }
}

template <class Reserve, class Inputs, class Solve>
void TestMarginalAccelerator(const char *name, bool available, Reserve reserve,
                             Inputs inputs, Solve solve) {
  if (!available)
    return;
  const btrc::Plan plan =
      btrc::MakePlan(std::vector<std::int64_t>{-1, 0, 1, 1, 0, 4, 5, 5});
  constexpr std::size_t kStates = 4;
  constexpr std::size_t kBatch = 5;
  std::vector<float> nodes(kBatch * plan.num_nodes() * kStates);
  for (std::size_t batch = 0; batch < kBatch; ++batch) {
    for (std::size_t node = 0; node < plan.num_nodes(); ++node) {
      for (std::size_t state = 0; state < kStates; ++state) {
        const std::size_t index =
            (batch * plan.num_nodes() + node) * kStates + state;
        nodes[index] = (batch + 2 * node + 3 * state) % 13 == 0
                           ? 0.0f
                           : 0.17f + 0.011f * static_cast<float>(
                                         1 + batch + 2 * node + 5 * state);
      }
    }
  }
  std::vector<float> edges(plan.num_edges() * kStates * kStates);
  for (std::size_t edge = 0; edge < plan.num_edges(); ++edge) {
    for (std::size_t parent = 0; parent < kStates; ++parent) {
      for (std::size_t child = 0; child < kStates; ++child) {
        const std::size_t index =
            (edge * kStates + parent) * kStates + child;
        edges[index] = (edge + 3 * parent + 5 * child) % 17 == 0
                           ? 0.0f
                           : 0.09f + 0.008f * static_cast<float>(
                                         1 + edge + parent + 2 * child);
      }
    }
  }

  reserve(plan, kStates, kBatch);
  tree_hmm::MutableBatchedModelView staged = inputs(kBatch);
  std::copy(nodes.begin(), nodes.end(), staged.node_potentials.begin());
  std::copy(edges.begin(), edges.end(), staged.edge_potentials.begin());
  const tree_hmm::BatchedMarginalView actual =
      solve(static_cast<tree_hmm::BatchedModelView>(staged));
  const std::size_t nodes_per_batch = plan.num_nodes() * kStates;
  const std::size_t edges_per_batch =
      plan.num_edges() * kStates * kStates;
  if (actual.log_partitions.size() != kBatch ||
      actual.nodes.size() != kBatch * nodes_per_batch ||
      actual.edges.size() != kBatch * edges_per_batch) {
    throw std::runtime_error(std::string(name) +
                             " marginal output shape is wrong");
  }
  for (std::size_t batch = 0; batch < kBatch; ++batch) {
    const std::vector<double> host_nodes(
        nodes.begin() + batch * nodes_per_batch,
        nodes.begin() + (batch + 1) * nodes_per_batch);
    const std::vector<double> host_edges(edges.begin(), edges.end());
    const tree_hmm::Marginals expected = tree_hmm::PosteriorMarginals(
        {plan, kStates, host_nodes, host_edges});
    if (std::abs(actual.log_partitions[batch] - expected.log_partition) >
        1e-4 * std::max(1.0, std::abs(expected.log_partition))) {
      throw std::runtime_error(std::string(name) +
                               " marginal log partition disagrees with CPU");
    }
    for (std::size_t index = 0; index < nodes_per_batch; ++index) {
      if (std::abs(actual.nodes[batch * nodes_per_batch + index] -
                   expected.nodes[index]) > 8e-5) {
        throw std::runtime_error(std::string(name) +
                                 " node marginal disagrees with CPU");
      }
    }
    for (std::size_t index = 0; index < edges_per_batch; ++index) {
      if (std::abs(actual.edges[batch * edges_per_batch + index] -
                   expected.edges[index]) > 8e-5) {
        throw std::runtime_error(std::string(name) +
                                 " edge marginal disagrees with CPU");
      }
    }
    for (std::size_t node = 0; node < plan.num_nodes(); ++node) {
      float total = 0.0f;
      for (std::size_t state = 0; state < kStates; ++state) {
        total += actual.nodes[(batch * plan.num_nodes() + node) * kStates +
                              state];
      }
      if (std::abs(total - 1.0f) > 1e-4f) {
        throw std::runtime_error(std::string(name) +
                                 " node marginal is not normalized");
      }
    }
  }

  const btrc::Plan root_plan =
      btrc::MakePlan(std::vector<std::int64_t>{-1});
  constexpr std::size_t kRootBatch = 3;
  constexpr std::size_t kRootStates = 3;
  const std::vector<float> root_nodes{
      0.2f, 0.3f, 0.5f, 0.1f, 0.0f, 0.9f, 0.4f, 0.4f, 0.2f};
  const std::vector<double> root_edges;
  reserve(root_plan, kRootStates, kRootBatch);
  tree_hmm::MutableBatchedModelView root_staged = inputs(kRootBatch);
  std::copy(root_nodes.begin(), root_nodes.end(),
            root_staged.node_potentials.begin());
  const tree_hmm::BatchedMarginalView root_result =
      solve(static_cast<tree_hmm::BatchedModelView>(root_staged));
  for (std::size_t batch = 0; batch < kRootBatch; ++batch) {
    const std::vector<double> host_nodes(
        root_nodes.begin() + batch * kRootStates,
        root_nodes.begin() + (batch + 1) * kRootStates);
    const tree_hmm::Marginals expected = tree_hmm::PosteriorMarginals(
        {root_plan, kRootStates, host_nodes, root_edges});
    for (std::size_t state = 0; state < kRootStates; ++state) {
      if (std::abs(root_result.nodes[batch * kRootStates + state] -
                   expected.nodes[state]) > 5e-6) {
        throw std::runtime_error(std::string(name) +
                                 " root-only marginal disagrees with CPU");
      }
    }
  }
}

#endif // TREE_HMM_TESTS_ACCELERATOR_TEST_H_
