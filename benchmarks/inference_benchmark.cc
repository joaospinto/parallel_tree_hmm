#include "tree_hmm/inference.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

double Median(std::vector<double> values) {
  std::sort(values.begin(), values.end());
  return values[values.size() / 2];
}

template <class Operation>
double Time(int repeats, Operation operation, double *checksum) {
  std::vector<double> times;
  times.reserve(repeats);
  for (int repeat = 0; repeat < repeats; ++repeat) {
    const Clock::time_point begin = Clock::now();
    *checksum += operation();
    const Clock::time_point end = Clock::now();
    times.push_back(
        std::chrono::duration<double, std::milli>(end - begin).count());
  }
  return Median(std::move(times));
}

btrc::Plan MakePlan(std::size_t leaves, std::string_view topology) {
  if (leaves == 0 || (leaves & (leaves - 1)) != 0)
    throw std::invalid_argument("leaf count must be a power of two");
  const std::size_t nodes = 2 * leaves - 1;
  std::vector<std::int64_t> parents(nodes, -1);
  if (topology == "balanced") {
    for (std::size_t node = 1; node < nodes; ++node)
      parents[node] = static_cast<std::int64_t>((node - 1) / 2);
  } else if (topology == "caterpillar") {
    const std::size_t internal_nodes = leaves - 1;
    for (std::size_t node = 1; node < internal_nodes; ++node)
      parents[node] = static_cast<std::int64_t>(node - 1);
    for (std::size_t leaf = 0; leaf + 2 < leaves; ++leaf)
      parents[internal_nodes + leaf] = static_cast<std::int64_t>(leaf);
    parents[nodes - 2] = static_cast<std::int64_t>(internal_nodes - 1);
    parents[nodes - 1] = static_cast<std::int64_t>(internal_nodes - 1);
  } else {
    throw std::invalid_argument("topology must be balanced or caterpillar");
  }
  return btrc::MakePlan(parents);
}

} // namespace

int main(int argc, char **argv) {
  try {
    const std::size_t leaves =
        argc > 1 ? static_cast<std::size_t>(std::strtoull(argv[1], nullptr, 10))
                 : 4096;
    const int repeats = argc > 2 ? std::max(1, std::atoi(argv[2])) : 11;
    const std::string_view topology = argc > 3 ? argv[3] : "balanced";
    constexpr std::size_t kStates = 4;
    const btrc::Plan plan = MakePlan(leaves, topology);
    std::vector<double> nodes(plan.num_nodes() * kStates);
    for (std::size_t index = 0; index < nodes.size(); ++index)
      nodes[index] = 0.2 + 0.01 * static_cast<double>(index % 17);
    std::vector<double> edges(plan.num_edges() * kStates * kStates);
    for (std::size_t edge = 0; edge < plan.num_edges(); ++edge) {
      for (std::size_t parent = 0; parent < kStates; ++parent) {
        for (std::size_t child = 0; child < kStates; ++child) {
          edges[(edge * kStates + parent) * kStates + child] =
              parent == child ? 0.85 : 0.05;
        }
      }
    }
    const tree_hmm::ModelView model{plan, kStates, nodes, edges};
    std::vector<double> uniforms(plan.num_nodes(), 0.5);
    tree_hmm::Workspace workspace;
    workspace.Reserve(plan, kStates);
    static_cast<void>(tree_hmm::LogPartitionFunctionPrepared(model, workspace));
    static_cast<void>(tree_hmm::PosteriorMarginalsPrepared(model, workspace));
    static_cast<void>(tree_hmm::MaximumAPosterioriPrepared(model, workspace));
    static_cast<void>(
        tree_hmm::PosteriorSamplePrepared(model, uniforms, workspace));

    double checksum = 0.0;
    const double log_partition_ms = Time(
        repeats,
        [&] {
          return tree_hmm::LogPartitionFunctionPrepared(model, workspace);
        },
        &checksum);
    const double marginals_ms = Time(
        repeats,
        [&] {
          return tree_hmm::PosteriorMarginalsPrepared(model, workspace)
              .log_partition;
        },
        &checksum);
    const double maximum_ms = Time(
        repeats,
        [&] {
          return tree_hmm::MaximumAPosterioriPrepared(model, workspace)
              .log_weight;
        },
        &checksum);
    const double sample_ms = Time(
        repeats,
        [&] {
          return static_cast<double>(
              tree_hmm::PosteriorSamplePrepared(model, uniforms, workspace)
                  .front());
        },
        &checksum);
    std::cout << std::setprecision(10)
              << "topology,leaves,nodes,states,repeats,log_partition_ms,"
                 "marginals_ms,map_ms,sample_ms,checksum\n"
              << topology << ',' << leaves << ',' << plan.num_nodes() << ','
              << kStates << ',' << repeats << ',' << log_partition_ms << ','
              << marginals_ms << ',' << maximum_ms << ',' << sample_ms << ','
              << checksum << '\n';
    return 0;
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
