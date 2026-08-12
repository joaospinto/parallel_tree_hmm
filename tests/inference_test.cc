#include "tree_hmm/inference.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <new>
#include <stdexcept>
#include <vector>

namespace {
bool g_count_allocations = false;
std::size_t g_allocations = 0;
}

void *operator new(std::size_t size) {
  if (g_count_allocations)
    ++g_allocations;
  if (void *result = std::malloc(size))
    return result;
  throw std::bad_alloc();
}

void operator delete(void *pointer) noexcept { std::free(pointer); }
void operator delete(void *pointer, std::size_t) noexcept { std::free(pointer); }

namespace {

bool Near(double left, double right, double tolerance = 1e-11) {
  return std::abs(left - right) <=
         tolerance * std::max({1.0, std::abs(left), std::abs(right)});
}

void Check(bool condition) {
  if (!condition)
    throw std::runtime_error("tree-HMM inference test failed");
}

tree_hmm::Marginals BruteForce(
    const btrc::Plan &plan, std::size_t states,
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
  return result;
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
      0.8, 0.2, 0.3, 0.7, 0.9, 0.1, 0.25, 0.75,
      0.6, 0.4, 0.15, 0.85, 0.7, 0.3, 0.2, 0.8,
  };
  const tree_hmm::ModelView model{plan, kStates, nodes, edges};
  const tree_hmm::Marginals expected =
      BruteForce(plan, kStates, nodes, edges);
  const tree_hmm::Marginals actual = tree_hmm::PosteriorMarginals(model);
  Check(Near(tree_hmm::PartitionFunction(model), expected.partition));
  Check(Near(tree_hmm::LogPartitionFunction(model),
             std::log(expected.partition)));
  Check(Near(actual.partition, expected.partition));
  for (std::size_t index = 0; index < actual.nodes.size(); ++index)
    Check(Near(actual.nodes[index], expected.nodes[index]));
  for (std::size_t index = 0; index < actual.edges.size(); ++index)
    Check(Near(actual.edges[index], expected.edges[index]));

  tree_hmm::Workspace workspace;
  workspace.Reserve(plan, kStates);
  const tree_hmm::MarginalView prepared =
      tree_hmm::PosteriorMarginalsPrepared(model, workspace);
  Check(Near(prepared.partition, expected.partition));
  g_allocations = 0;
  g_count_allocations = true;
  for (int repeat = 0; repeat < 10; ++repeat) {
    const tree_hmm::MarginalView repeated =
        tree_hmm::PosteriorMarginalsPrepared(model, workspace);
    Check(Near(repeated.partition, expected.partition));
    Check(Near(tree_hmm::LogPartitionFunctionPrepared(model, workspace),
               std::log(expected.partition)));
  }
  g_count_allocations = false;
  Check(g_allocations == 0);

  constexpr std::size_t kLongNodes = 4096;
  std::vector<std::int64_t> long_parents(kLongNodes);
  long_parents[0] = -1;
  for (std::size_t node = 1; node < kLongNodes; ++node)
    long_parents[node] = static_cast<std::int64_t>(node - 1);
  const btrc::Plan long_plan = btrc::MakePlan(long_parents);
  const std::vector<double> long_nodes(kLongNodes, 0.5);
  const std::vector<double> long_edges(kLongNodes - 1, 1.0);
  const double log_partition = tree_hmm::LogPartitionFunction(
      {long_plan, 1, long_nodes, long_edges});
  Check(Near(log_partition, kLongNodes * std::log(0.5), 1e-10));
}
