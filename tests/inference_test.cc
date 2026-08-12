#include "tree_hmm/inference.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <new>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
bool g_count_allocations = false;
std::size_t g_allocations = 0;
} // namespace

void *operator new(std::size_t size) {
  if (g_count_allocations)
    ++g_allocations;
  if (void *result = std::malloc(size))
    return result;
  throw std::bad_alloc();
}

void operator delete(void *pointer) noexcept { std::free(pointer); }
void operator delete(void *pointer, std::size_t) noexcept {
  std::free(pointer);
}

namespace {

bool Near(double left, double right, double tolerance = 1e-11) {
  return std::abs(left - right) <=
         tolerance * std::max({1.0, std::abs(left), std::abs(right)});
}

void CheckImpl(bool condition, int line) {
  if (!condition)
    throw std::runtime_error("tree-HMM inference test failed at line " +
                             std::to_string(line));
}

#define Check(condition) CheckImpl((condition), __LINE__)

tree_hmm::Marginals BruteForce(const btrc::Plan &plan, std::size_t states,
                               const std::vector<double> &node_potentials,
                               const std::vector<double> &edge_potentials) {
  tree_hmm::Marginals result;
  result.nodes.assign(node_potentials.size(), 0.0);
  result.edges.assign(edge_potentials.size(), 0.0);
  std::vector<std::size_t> assignment(plan.num_nodes(), 0);
  std::size_t assignments = 1;
  for (std::size_t node = 0; node < plan.num_nodes(); ++node)
    assignments *= states;

  for (std::size_t code = 0; code < assignments; ++code) {
    std::size_t remainder = code;
    double weight = 1.0;
    for (std::size_t node = 0; node < plan.num_nodes(); ++node) {
      assignment[node] = remainder % states;
      remainder /= states;
      weight *= node_potentials[node * states + assignment[node]];
    }
    for (std::size_t edge = 0; edge < plan.num_edges(); ++edge) {
      const std::size_t parent_state = assignment[plan.edge_parents()[edge]];
      const std::size_t child_state = assignment[plan.edge_children()[edge]];
      weight *= edge_potentials[(edge * states + parent_state) * states +
                                child_state];
    }
    result.partition += weight;
    for (std::size_t node = 0; node < plan.num_nodes(); ++node)
      result.nodes[node * states + assignment[node]] += weight;
    for (std::size_t edge = 0; edge < plan.num_edges(); ++edge) {
      const std::size_t parent_state = assignment[plan.edge_parents()[edge]];
      const std::size_t child_state = assignment[plan.edge_children()[edge]];
      result.edges[(edge * states + parent_state) * states + child_state] +=
          weight;
    }
  }
  for (double &value : result.nodes)
    value /= result.partition;
  for (double &value : result.edges)
    value /= result.partition;
  result.log_partition = std::log(result.partition);
  return result;
}

double AssignmentWeight(const btrc::Plan &plan, std::size_t states,
                        const std::vector<double> &node_potentials,
                        const std::vector<double> &edge_potentials,
                        std::span<const std::size_t> assignment) {
  Check(assignment.size() == plan.num_nodes());
  double result = 1.0;
  for (std::size_t node = 0; node < plan.num_nodes(); ++node) {
    Check(assignment[node] < states);
    result *= node_potentials[node * states + assignment[node]];
  }
  for (std::size_t edge = 0; edge < plan.num_edges(); ++edge) {
    const std::size_t parent_state = assignment[plan.edge_parents()[edge]];
    const std::size_t child_state = assignment[plan.edge_children()[edge]];
    result *=
        edge_potentials[(edge * states + parent_state) * states + child_state];
  }
  return result;
}

tree_hmm::MaximumAssignment
BruteForceMaximum(const btrc::Plan &plan, std::size_t states,
                  const std::vector<double> &node_potentials,
                  const std::vector<double> &edge_potentials) {
  tree_hmm::MaximumAssignment result;
  result.states.assign(plan.num_nodes(), 0);
  std::vector<std::size_t> assignment(plan.num_nodes(), 0);
  std::size_t assignments = 1;
  for (std::size_t node = 0; node < plan.num_nodes(); ++node)
    assignments *= states;
  for (std::size_t code = 0; code < assignments; ++code) {
    std::size_t remainder = code;
    for (std::size_t node = 0; node < plan.num_nodes(); ++node) {
      assignment[node] = remainder % states;
      remainder /= states;
    }
    const double weight = AssignmentWeight(plan, states, node_potentials,
                                           edge_potentials, assignment);
    if (weight > result.weight) {
      result.weight = weight;
      result.states = assignment;
    }
  }
  result.log_weight = std::log(result.weight);
  return result;
}

void CheckMarginals(const tree_hmm::Marginals &actual,
                    const tree_hmm::Marginals &expected) {
  Check(Near(actual.partition, expected.partition));
  Check(Near(actual.log_partition, expected.log_partition));
  for (std::size_t index = 0; index < actual.nodes.size(); ++index)
    Check(Near(actual.nodes[index], expected.nodes[index]));
  for (std::size_t index = 0; index < actual.edges.size(); ++index)
    Check(Near(actual.edges[index], expected.edges[index]));
}

} // namespace

int main() {
  const btrc::Plan plan =
      btrc::MakePlan(std::vector<std::int64_t>{-1, 0, 0, 1, 3});
  constexpr std::size_t kStates = 2;
  const std::vector<double> nodes{
      0.55, 0.45, 0.9, 0.2, 0.4, 0.8, 0.7, 0.3, 0.1, 0.95,
  };
  const std::vector<double> edges{
      0.8, 0.2, 0.3,  0.7,  0.9, 0.1, 0.25, 0.75,
      0.6, 0.4, 0.15, 0.85, 0.7, 0.3, 0.2,  0.8,
  };
  const tree_hmm::ModelView model{plan, kStates, nodes, edges};
  const tree_hmm::Marginals expected = BruteForce(plan, kStates, nodes, edges);
  const tree_hmm::Marginals actual = tree_hmm::PosteriorMarginals(model);
  Check(Near(tree_hmm::PartitionFunction(model), expected.partition));
  Check(Near(tree_hmm::LogPartitionFunction(model),
             std::log(expected.partition)));
  CheckMarginals(actual, expected);
  Check(plan.num_compressions() != 0);

  const tree_hmm::MaximumAssignment expected_maximum =
      BruteForceMaximum(plan, kStates, nodes, edges);
  const tree_hmm::MaximumAssignment maximum =
      tree_hmm::MaximumAPosteriori(model);
  Check(Near(maximum.weight, expected_maximum.weight));
  Check(Near(maximum.log_weight, expected_maximum.log_weight));
  Check(Near(AssignmentWeight(plan, kStates, nodes, edges, maximum.states),
             expected_maximum.weight));

  const std::vector<double> zero_nodes{
      0.55, 0.45, 1.0, 0.0, 0.0, 1.0, 0.7, 0.3, 0.0, 1.0,
  };
  const std::vector<double> zero_edges{
      0.8, 0.2, 0.3, 0.7, 1.0, 0.0, 0.25, 0.75,
      0.6, 0.4, 0.0, 1.0, 0.7, 0.3, 0.2,  0.8,
  };
  const tree_hmm::Marginals zero_expected =
      BruteForce(plan, kStates, zero_nodes, zero_edges);
  CheckMarginals(
      tree_hmm::PosteriorMarginals({plan, kStates, zero_nodes, zero_edges}),
      zero_expected);
  const tree_hmm::MaximumAssignment zero_maximum =
      tree_hmm::MaximumAPosteriori({plan, kStates, zero_nodes, zero_edges});
  Check(Near(zero_maximum.weight,
             BruteForceMaximum(plan, kStates, zero_nodes, zero_edges).weight));

  const std::vector<double> midpoint_uniforms(plan.num_nodes(), 0.5);
  const std::vector<std::size_t> zero_sample = tree_hmm::PosteriorSample(
      {plan, kStates, zero_nodes, zero_edges}, midpoint_uniforms);
  Check(AssignmentWeight(plan, kStates, zero_nodes, zero_edges, zero_sample) >
        0.0);

  const std::vector<double> tie_nodes(plan.num_nodes() * kStates, 1.0);
  const std::vector<double> tie_edges(plan.num_edges() * kStates * kStates,
                                      1.0);
  const tree_hmm::MaximumAssignment tie_maximum =
      tree_hmm::MaximumAPosteriori({plan, kStates, tie_nodes, tie_edges});
  Check(std::all_of(tie_maximum.states.begin(), tie_maximum.states.end(),
                    [](std::size_t state) { return state == 0; }));

  tree_hmm::Workspace workspace;
  workspace.Reserve(plan, kStates);
  const tree_hmm::MarginalView prepared =
      tree_hmm::PosteriorMarginalsPrepared(model, workspace);
  Check(Near(prepared.partition, expected.partition));
  Check(Near(prepared.log_partition, std::log(expected.partition)));
  const tree_hmm::MaximumAssignmentView prepared_maximum =
      tree_hmm::MaximumAPosterioriPrepared(model, workspace);
  Check(Near(prepared_maximum.weight, expected_maximum.weight));
  std::vector<double> uniforms(plan.num_nodes(), 0.5);
  const std::span<const std::size_t> prepared_sample =
      tree_hmm::PosteriorSamplePrepared(model, uniforms, workspace);
  Check(AssignmentWeight(plan, kStates, nodes, edges, prepared_sample) > 0.0);
  g_allocations = 0;
  g_count_allocations = true;
  for (int repeat = 0; repeat < 10; ++repeat) {
    const tree_hmm::MarginalView repeated =
        tree_hmm::PosteriorMarginalsPrepared(model, workspace);
    Check(Near(repeated.partition, expected.partition));
    Check(Near(tree_hmm::LogPartitionFunctionPrepared(model, workspace),
               std::log(expected.partition)));
    Check(Near(tree_hmm::MaximumAPosterioriPrepared(model, workspace).weight,
               expected_maximum.weight));
    Check(
        tree_hmm::PosteriorSamplePrepared(model, uniforms, workspace).size() ==
        plan.num_nodes());
  }
  g_count_allocations = false;
  Check(g_allocations == 0);

  constexpr std::size_t kSamples = 30000;
  std::vector<double> sampled_nodes(expected.nodes.size(), 0.0);
  std::vector<double> sampled_edges(expected.edges.size(), 0.0);
  std::uint64_t random_state = 0xd1b54a32d192ed03ULL;
  const auto next_uniform = [&] {
    random_state =
        random_state * 6364136223846793005ULL + 1442695040888963407ULL;
    return static_cast<double>(random_state >> 11) * 0x1.0p-53;
  };
  for (std::size_t draw = 0; draw < kSamples; ++draw) {
    for (double &uniform : uniforms)
      uniform = next_uniform();
    const std::span<const std::size_t> sample =
        tree_hmm::PosteriorSamplePrepared(model, uniforms, workspace);
    for (std::size_t node = 0; node < plan.num_nodes(); ++node)
      ++sampled_nodes[node * kStates + sample[node]];
    for (std::size_t edge = 0; edge < plan.num_edges(); ++edge) {
      const std::size_t parent_state = sample[plan.edge_parents()[edge]];
      const std::size_t child_state = sample[plan.edge_children()[edge]];
      ++sampled_edges[(edge * kStates + parent_state) * kStates + child_state];
    }
  }
  for (std::size_t index = 0; index < sampled_nodes.size(); ++index)
    Check(Near(sampled_nodes[index] / kSamples, expected.nodes[index], 0.012));
  for (std::size_t index = 0; index < sampled_edges.size(); ++index)
    Check(Near(sampled_edges[index] / kSamples, expected.edges[index], 0.012));

  bool invalid_uniform_rejected = false;
  uniforms[0] = 1.0;
  try {
    static_cast<void>(
        tree_hmm::PosteriorSamplePrepared(model, uniforms, workspace));
  } catch (const std::invalid_argument &) {
    invalid_uniform_rejected = true;
  }
  Check(invalid_uniform_rejected);

  constexpr std::size_t kLongNodes = 4096;
  std::vector<std::int64_t> long_parents(kLongNodes);
  long_parents[0] = -1;
  for (std::size_t node = 1; node < kLongNodes; ++node)
    long_parents[node] = static_cast<std::int64_t>(node - 1);
  const btrc::Plan long_plan = btrc::MakePlan(long_parents);
  const std::vector<double> long_nodes(kLongNodes, 0.5);
  const std::vector<double> long_edges(kLongNodes - 1, 1.0);
  const double log_partition =
      tree_hmm::LogPartitionFunction({long_plan, 1, long_nodes, long_edges});
  Check(Near(log_partition, kLongNodes * std::log(0.5), 1e-10));
  const tree_hmm::Marginals long_marginals =
      tree_hmm::PosteriorMarginals({long_plan, 1, long_nodes, long_edges});
  Check(long_marginals.partition == 0.0);
  Check(Near(long_marginals.log_partition, log_partition, 1e-10));
  Check(std::all_of(long_marginals.nodes.begin(), long_marginals.nodes.end(),
                    [](double value) { return Near(value, 1.0); }));
  Check(std::all_of(long_marginals.edges.begin(), long_marginals.edges.end(),
                    [](double value) { return Near(value, 1.0); }));
  const tree_hmm::MaximumAssignment long_maximum =
      tree_hmm::MaximumAPosteriori({long_plan, 1, long_nodes, long_edges});
  Check(long_maximum.weight == 0.0);
  Check(Near(long_maximum.log_weight, log_partition, 1e-10));

  const std::vector<double> long_binary_nodes(kLongNodes * 2, 0.5);
  std::vector<double> long_binary_edges((kLongNodes - 1) * 4);
  for (std::size_t edge = 0; edge < kLongNodes - 1; ++edge) {
    std::copy_n(std::array<double, 4>{0.9, 0.1, 0.2, 0.8}.begin(), 4,
                long_binary_edges.begin() + edge * 4);
  }
  const tree_hmm::Marginals long_binary_marginals =
      tree_hmm::PosteriorMarginals(
          {long_plan, 2, long_binary_nodes, long_binary_edges});
  Check(long_binary_marginals.partition == 0.0);
  Check(Near(long_binary_marginals.log_partition,
             (kLongNodes - 1) * std::log(0.5), 1e-10));
  for (std::size_t node = 0; node < kLongNodes; ++node) {
    Check(Near(long_binary_marginals.nodes[node * 2] +
                   long_binary_marginals.nodes[node * 2 + 1],
               1.0));
  }
  for (std::size_t edge = 0; edge < kLongNodes - 1; ++edge) {
    const double *joint = long_binary_marginals.edges.data() + edge * 4;
    Check(Near(std::accumulate(joint, joint + 4, 0.0), 1.0));
    const btrc::Index parent = long_plan.edge_parents()[edge];
    const btrc::Index child = long_plan.edge_children()[edge];
    Check(Near(joint[0] + joint[1], long_binary_marginals.nodes[parent * 2]));
    Check(
        Near(joint[2] + joint[3], long_binary_marginals.nodes[parent * 2 + 1]));
    Check(Near(joint[0] + joint[2], long_binary_marginals.nodes[child * 2]));
    Check(
        Near(joint[1] + joint[3], long_binary_marginals.nodes[child * 2 + 1]));
  }
}
