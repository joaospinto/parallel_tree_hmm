#include "src/cuda_device_algebra.h"

#include "btrc/execute.h"
#include "tree_hmm/inference.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <span>
#include <stdexcept>
#include <vector>

namespace {

class EmulatedDispatcher {
public:
  EmulatedDispatcher(const btrc::Plan &plan, std::size_t states,
                     std::span<const float> nodes,
                     std::span<const float> paths, bool scaled)
      : plan_(plan), states_(states), matrix_(states * states),
        nodes_(nodes.begin(), nodes.end()), paths_(paths.begin(), paths.end()),
        branches_(plan.num_branches() * states),
        node_scales_(plan.num_nodes()), path_scales_(plan.num_edges()),
        branch_scales_(plan.num_branches()), scratch_(matrix_), scaled_(scaled) {
  }

  void Rake(std::span<const btrc::Rake> operations) {
    for (const btrc::Rake &operation : operations) {
      float *output = Branch(operation.branch);
      for (std::size_t state = 0; state < states_; ++state) {
        output[state] = tree_hmm::cuda::detail::RakeValue(
            Path(operation.edge), Node(operation.leaf), states_, state);
      }
      if (scaled_) {
        BranchScale(operation.branch) =
            Normalize(output, states_, PathScale(operation.edge) +
                                            NodeScale(operation.leaf));
      }
    }
  }

  void CombineBranches(std::span<const btrc::BranchCombination> operations) {
    for (const btrc::BranchCombination &operation : operations) {
      float *destination = Branch(operation.destination);
      const float *source = Branch(operation.source);
      for (std::size_t state = 0; state < states_; ++state) {
        destination[state] = tree_hmm::cuda::detail::Product(
            destination[state], source[state]);
      }
      if (scaled_) {
        BranchScale(operation.destination) = Normalize(
            destination, states_, BranchScale(operation.destination) +
                                      BranchScale(operation.source));
      }
    }
  }

  void AbsorbBranches(std::span<const btrc::BranchAbsorption> operations) {
    for (const btrc::BranchAbsorption &operation : operations) {
      float *node = Node(operation.parent);
      const float *branch = Branch(operation.branch);
      for (std::size_t state = 0; state < states_; ++state) {
        node[state] =
            tree_hmm::cuda::detail::Product(node[state], branch[state]);
      }
      if (scaled_) {
        NodeScale(operation.parent) = Normalize(
            node, states_, NodeScale(operation.parent) +
                               BranchScale(operation.branch));
      }
    }
  }

  void Compress(std::span<const btrc::Compression> operations) {
    for (const btrc::Compression &operation : operations) {
      float *left = Path(operation.left_edge);
      const float *middle = Node(operation.middle);
      const float *right = Path(operation.right_edge);
      std::copy(left, left + matrix_, scratch_.begin());
      for (std::size_t parent = 0; parent < states_; ++parent) {
        for (std::size_t child = 0; child < states_; ++child) {
          left[parent * states_ + child] =
              tree_hmm::cuda::detail::CompressionValue(
                  scratch_.data(), middle, right, states_, parent, child);
        }
      }
      if (scaled_) {
        PathScale(operation.left_edge) = Normalize(
            left, matrix_, PathScale(operation.left_edge) +
                               NodeScale(operation.middle) +
                               PathScale(operation.right_edge));
      }
    }
  }

  float Finish() const {
    const float sum =
        std::accumulate(Node(plan_.root()), Node(plan_.root()) + states_, 0.0f);
    return scaled_ ? NodeScale(plan_.root()) + std::log(sum) : sum;
  }

private:
  float Normalize(float *values, std::size_t size, float input_scale) const {
    const float maximum = tree_hmm::cuda::detail::Maximum(
        values, static_cast<unsigned>(size));
    if (maximum > 0.0f) {
      for (std::size_t index = 0; index < size; ++index)
        values[index] /= maximum;
    }
    return tree_hmm::cuda::detail::UpdatedLogScale(input_scale, maximum);
  }
  float *Node(btrc::Index index) { return nodes_.data() + index * states_; }
  const float *Node(btrc::Index index) const {
    return nodes_.data() + index * states_;
  }
  float *Path(btrc::Index index) { return paths_.data() + index * matrix_; }
  const float *Path(btrc::Index index) const {
    return paths_.data() + index * matrix_;
  }
  float *Branch(btrc::Index index) {
    return branches_.data() + index * states_;
  }
  const float *Branch(btrc::Index index) const {
    return branches_.data() + index * states_;
  }
  float &NodeScale(btrc::Index index) { return node_scales_[index]; }
  float NodeScale(btrc::Index index) const { return node_scales_[index]; }
  float &PathScale(btrc::Index index) { return path_scales_[index]; }
  float PathScale(btrc::Index index) const { return path_scales_[index]; }
  float &BranchScale(btrc::Index index) { return branch_scales_[index]; }
  float BranchScale(btrc::Index index) const { return branch_scales_[index]; }

  const btrc::Plan &plan_;
  std::size_t states_;
  std::size_t matrix_;
  std::vector<float> nodes_;
  std::vector<float> paths_;
  std::vector<float> branches_;
  std::vector<float> node_scales_;
  std::vector<float> path_scales_;
  std::vector<float> branch_scales_;
  std::vector<float> scratch_;
  bool scaled_;
};

bool Near(float left, double right, double tolerance = 5e-5) {
  return std::abs(static_cast<double>(left) - right) <=
         tolerance * std::max({1.0, std::abs(static_cast<double>(left)),
                               std::abs(right)});
}

} // namespace

int main() {
  const btrc::Plan plan =
      btrc::MakePlan(std::vector<std::int64_t>{-1, 0, 0, 0, 1, 1, 2, 6});
  constexpr std::size_t kStates = 4;
  std::vector<float> nodes(plan.num_nodes() * kStates);
  for (std::size_t index = 0; index < nodes.size(); ++index)
    nodes[index] = 0.15f + 0.01f * static_cast<float>(index % 11);
  std::vector<float> edges(plan.num_edges() * kStates * kStates);
  for (std::size_t edge = 0; edge < plan.num_edges(); ++edge) {
    for (std::size_t parent = 0; parent < kStates; ++parent) {
      for (std::size_t child = 0; child < kStates; ++child) {
        edges[(edge * kStates + parent) * kStates + child] =
            parent == child ? 0.85f : 0.05f;
      }
    }
  }
  const std::vector<double> host_nodes(nodes.begin(), nodes.end());
  const std::vector<double> host_edges(edges.begin(), edges.end());
  const tree_hmm::ModelView host_model{plan, kStates, host_nodes, host_edges};

  EmulatedDispatcher raw(plan, kStates, nodes, edges, false);
  btrc::Contract(plan, raw);
  if (!Near(raw.Finish(), tree_hmm::PartitionFunction(host_model)))
    throw std::runtime_error("emulated CUDA partition function is incorrect");

  EmulatedDispatcher scaled(plan, kStates, nodes, edges, true);
  btrc::Contract(plan, scaled);
  if (!Near(scaled.Finish(), tree_hmm::LogPartitionFunction(host_model)))
    throw std::runtime_error("emulated CUDA log partition is incorrect");
}
