#include "src/gpu_device_algebra.h"

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

class HostAlgebraDispatcher {
public:
  HostAlgebraDispatcher(const btrc::Plan &plan, std::size_t states,
                        std::span<const tree_hmm::Scalar> nodes,
                        std::span<const tree_hmm::Scalar> paths, bool scaled)
      : plan_(plan), states_(states), matrix_(states * states),
        nodes_(nodes.begin(), nodes.end()), paths_(paths.begin(), paths.end()),
        branches_(plan.num_branches() * states), node_scales_(plan.num_nodes()),
        path_scales_(plan.num_edges()), branch_scales_(plan.num_branches()),
        scratch_(matrix_), weighted_right_(matrix_), scaled_(scaled) {}

  void Rake(std::span<const btrc::Rake> operations) {
    for (const btrc::Rake &operation : operations) {
      tree_hmm::Scalar *output = Branch(operation.branch);
      for (std::size_t state = 0; state < states_; ++state) {
        output[state] = tree_hmm::accelerator_detail::RakeValue(
            Path(operation.edge), Node(operation.leaf), states_, state);
      }
      if (scaled_) {
        BranchScale(operation.branch) =
            Normalize(output, states_,
                      PathScale(operation.edge) + NodeScale(operation.leaf));
      }
    }
  }

  void CombineBranches(std::span<const btrc::BranchCombination> operations) {
    for (const btrc::BranchCombination &operation : operations) {
      tree_hmm::Scalar *destination = Branch(operation.destination);
      const tree_hmm::Scalar *source = Branch(operation.source);
      for (std::size_t state = 0; state < states_; ++state) {
        destination[state] =
            tree_hmm::accelerator_detail::Product(destination[state], source[state]);
      }
      if (scaled_) {
        BranchScale(operation.destination) = Normalize(
            destination, states_,
            BranchScale(operation.destination) + BranchScale(operation.source));
      }
    }
  }

  void AbsorbBranches(std::span<const btrc::BranchAbsorption> operations) {
    for (const btrc::BranchAbsorption &operation : operations) {
      tree_hmm::Scalar *node = Node(operation.parent);
      const tree_hmm::Scalar *branch = Branch(operation.branch);
      for (std::size_t state = 0; state < states_; ++state) {
        node[state] =
            tree_hmm::accelerator_detail::Product(node[state], branch[state]);
      }
      if (scaled_) {
        NodeScale(operation.parent) = Normalize(
            node, states_,
            NodeScale(operation.parent) + BranchScale(operation.branch));
      }
    }
  }

  void Compress(std::span<const btrc::Compression> operations) {
    for (const btrc::Compression &operation : operations) {
      tree_hmm::Scalar *left = Path(operation.left_edge);
      const tree_hmm::Scalar *middle = Node(operation.middle);
      const tree_hmm::Scalar *right = Path(operation.right_edge);
      std::copy(left, left + matrix_, scratch_.begin());
      for (std::size_t entry = 0; entry < matrix_; ++entry)
        weighted_right_[entry] = right[entry] * middle[entry / states_];
      for (std::size_t parent = 0; parent < states_; ++parent) {
        for (std::size_t child = 0; child < states_; ++child) {
          left[parent * states_ + child] =
              tree_hmm::accelerator_detail::MatrixProductValue(
                  scratch_.data(), weighted_right_.data(), states_, parent,
                  child);
        }
      }
      if (scaled_) {
        PathScale(operation.left_edge) = Normalize(
            left, matrix_,
            PathScale(operation.left_edge) + NodeScale(operation.middle) +
                PathScale(operation.right_edge));
      }
    }
  }

  tree_hmm::Scalar Finish() const {
    const tree_hmm::Scalar sum =
        std::accumulate(Node(plan_.root()), Node(plan_.root()) + states_, 0.0f);
    return scaled_ ? NodeScale(plan_.root()) + std::log(sum) : sum;
  }

private:
  tree_hmm::Scalar Normalize(tree_hmm::Scalar *values, std::size_t size,
                             tree_hmm::Scalar input_scale) const {
    const tree_hmm::Scalar maximum =
        tree_hmm::accelerator_detail::Maximum(values, static_cast<unsigned>(size));
    if (maximum > 0.0f) {
      for (std::size_t index = 0; index < size; ++index)
        values[index] /= maximum;
    }
    return tree_hmm::accelerator_detail::UpdatedLogScale(input_scale, maximum);
  }
  tree_hmm::Scalar *Node(btrc::Index index) {
    return nodes_.data() + index * states_;
  }
  const tree_hmm::Scalar *Node(btrc::Index index) const {
    return nodes_.data() + index * states_;
  }
  tree_hmm::Scalar *Path(btrc::Index index) {
    return paths_.data() + index * matrix_;
  }
  const tree_hmm::Scalar *Path(btrc::Index index) const {
    return paths_.data() + index * matrix_;
  }
  tree_hmm::Scalar *Branch(btrc::Index index) {
    return branches_.data() + index * states_;
  }
  const tree_hmm::Scalar *Branch(btrc::Index index) const {
    return branches_.data() + index * states_;
  }
  tree_hmm::Scalar &NodeScale(btrc::Index index) { return node_scales_[index]; }
  tree_hmm::Scalar NodeScale(btrc::Index index) const {
    return node_scales_[index];
  }
  tree_hmm::Scalar &PathScale(btrc::Index index) { return path_scales_[index]; }
  tree_hmm::Scalar PathScale(btrc::Index index) const {
    return path_scales_[index];
  }
  tree_hmm::Scalar &BranchScale(btrc::Index index) {
    return branch_scales_[index];
  }
  tree_hmm::Scalar BranchScale(btrc::Index index) const {
    return branch_scales_[index];
  }

  const btrc::Plan &plan_;
  std::size_t states_;
  std::size_t matrix_;
  std::vector<tree_hmm::Scalar> nodes_;
  std::vector<tree_hmm::Scalar> paths_;
  std::vector<tree_hmm::Scalar> branches_;
  std::vector<tree_hmm::Scalar> node_scales_;
  std::vector<tree_hmm::Scalar> path_scales_;
  std::vector<tree_hmm::Scalar> branch_scales_;
  std::vector<tree_hmm::Scalar> scratch_;
  std::vector<tree_hmm::Scalar> weighted_right_;
  bool scaled_;
};

bool Near(tree_hmm::Scalar left, double right, double tolerance = 5e-5) {
  return std::abs(static_cast<double>(left) - right) <=
         tolerance * std::max({1.0, std::abs(static_cast<double>(left)),
                               std::abs(right)});
}

} // namespace

int main() {
  const btrc::Plan plan =
      btrc::MakePlan(std::vector<std::int64_t>{-1, 0, 0, 0, 1, 1, 2, 6});
  constexpr std::size_t kStates = 4;
  std::vector<tree_hmm::Scalar> nodes(plan.num_nodes() * kStates);
  for (std::size_t index = 0; index < nodes.size(); ++index)
    nodes[index] = 0.15f + 0.01f * static_cast<tree_hmm::Scalar>(index % 11);
  std::vector<tree_hmm::Scalar> edges(plan.num_edges() * kStates * kStates);
  for (std::size_t edge = 0; edge < plan.num_edges(); ++edge) {
    for (std::size_t parent = 0; parent < kStates; ++parent) {
      for (std::size_t child = 0; child < kStates; ++child) {
        edges[(edge * kStates + parent) * kStates + child] =
            parent == child ? 0.85f : 0.05f;
      }
    }
  }
  const std::vector<tree_hmm::Scalar> host_nodes(nodes.begin(), nodes.end());
  const std::vector<tree_hmm::Scalar> host_edges(edges.begin(), edges.end());
  const tree_hmm::ModelView host_model{plan, kStates, host_nodes, host_edges};

  HostAlgebraDispatcher raw(plan, kStates, nodes, edges, false);
  btrc::Contract(plan, raw);
  if (!Near(raw.Finish(), tree_hmm::PartitionFunction(host_model)))
    throw std::runtime_error(
        "host-side device-algebra partition function is incorrect");

  HostAlgebraDispatcher scaled(plan, kStates, nodes, edges, true);
  btrc::Contract(plan, scaled);
  if (!Near(scaled.Finish(), tree_hmm::LogPartitionFunction(host_model)))
    throw std::runtime_error(
        "host-side device-algebra log partition is incorrect");
}
