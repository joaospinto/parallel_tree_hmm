#include "tree_hmm/inference.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>

#include "btrc/execute.h"

namespace tree_hmm {

struct Workspace::Impl {
  const btrc::Plan *plan = nullptr;
  std::size_t states = 0;

  std::vector<Scalar> nodes;
  std::vector<Scalar> paths;
  std::vector<Scalar> branches;
  std::vector<Scalar> node_log_scales;
  std::vector<Scalar> path_log_scales;
  std::vector<Scalar> branch_log_scales;
  std::vector<Scalar> node_adjoints;
  std::vector<Scalar> path_adjoints;
  std::vector<Scalar> branch_adjoints;

  std::vector<Scalar> rake_paths;
  std::vector<Scalar> rake_leaves;
  std::vector<Scalar> compression_left;
  std::vector<Scalar> compression_middle;
  std::vector<Scalar> compression_right;
  std::vector<Scalar> reverse_scratch;

  std::vector<std::size_t> rake_choices;
  std::vector<std::size_t> compression_choices;
  std::vector<std::size_t> assignments;
  Scalar partition = 0.0;
  Scalar log_partition = 0.0;
  Scalar maximum_log_weight = 0.0;
};

namespace {

std::size_t CheckedProduct(std::size_t left, std::size_t right,
                           const char *description) {
  if (right != 0 && left > std::numeric_limits<std::size_t>::max() / right)
    throw std::length_error(std::string(description) + " overflows size_t");
  return left * right;
}

void Validate(ModelView model) {
  if (model.states == 0)
    throw std::invalid_argument("a tree HMM must have at least one state");
  const std::size_t matrix_size =
      CheckedProduct(model.states, model.states, "state matrix size");
  const std::size_t expected_nodes =
      CheckedProduct(model.plan.num_nodes(), model.states, "node potentials");
  const std::size_t expected_edges =
      CheckedProduct(model.plan.num_edges(), matrix_size, "edge potentials");
  if (model.node_potentials.size() != expected_nodes)
    throw std::invalid_argument("node-potential shape does not match the plan");
  if (model.edge_potentials.size() != expected_edges)
    throw std::invalid_argument("edge-potential shape does not match the plan");
  const auto invalid = [](Scalar value) {
    return !std::isfinite(value) || value < 0.0;
  };
  if (std::any_of(model.node_potentials.begin(), model.node_potentials.end(),
                  invalid) ||
      std::any_of(model.edge_potentials.begin(), model.edge_potentials.end(),
                  invalid)) {
    throw std::invalid_argument(
        "tree-HMM potentials must be finite and nonnegative");
  }
}

Workspace::Impl &Prepare(ModelView model, Workspace::Impl &storage) {
  Validate(model);
  if (storage.plan != &model.plan || storage.states != model.states) {
    throw std::invalid_argument(
        "prepared inference requires Workspace::Reserve for this plan and "
        "state count");
  }
  std::copy(model.node_potentials.begin(), model.node_potentials.end(),
            storage.nodes.begin());
  std::copy(model.edge_potentials.begin(), model.edge_potentials.end(),
            storage.paths.begin());
  storage.partition = 0.0;
  storage.log_partition = 0.0;
  storage.maximum_log_weight = 0.0;
  return storage;
}

Scalar PotentialLog(Scalar value) {
  return value > 0.0 ? std::log(value)
                     : -std::numeric_limits<Scalar>::infinity();
}

template <class Term> Scalar LogSumExp(std::size_t size, Term term) {
  Scalar maximum = -std::numeric_limits<Scalar>::infinity();
  for (std::size_t index = 0; index < size; ++index)
    maximum = std::max(maximum, term(index));
  if (!std::isfinite(maximum))
    return maximum;
  Scalar sum = 0.0;
  for (std::size_t index = 0; index < size; ++index)
    sum += std::exp(term(index) - maximum);
  return maximum + std::log(sum);
}

template <class Term>
std::size_t Argmax(std::size_t size, Term term, Scalar *maximum = nullptr) {
  std::size_t result = 0;
  Scalar value = term(0);
  for (std::size_t index = 1; index < size; ++index) {
    const Scalar candidate = term(index);
    if (candidate > value) {
      result = index;
      value = candidate;
    }
  }
  if (maximum != nullptr)
    *maximum = value;
  return result;
}

template <class Term>
std::size_t SampleLogWeights(std::size_t size, Term term, Scalar normalizer,
                             Scalar uniform) {
  Scalar cumulative = 0.0;
  std::size_t last_supported = size;
  for (std::size_t index = 0; index < size; ++index) {
    const Scalar log_weight = term(index);
    if (!std::isfinite(log_weight))
      continue;
    last_supported = index;
    cumulative += std::exp(log_weight - normalizer);
    if (uniform < cumulative)
      return index;
  }
  if (last_supported == size)
    throw std::domain_error(
        "cannot sample from an empty conditional distribution");
  return last_supported;
}

class SumProductDispatcher {
public:
  SumProductDispatcher(const btrc::Plan &plan, std::size_t states,
                       Workspace::Impl &storage)
      : plan_(plan), states_(states), matrix_size_(states * states),
        storage_(storage) {}

  void Rake(std::span<const btrc::Rake> operations) {
    for (const auto &operation : operations) {
      const Scalar *path = Path(operation.edge);
      const Scalar *leaf = Node(operation.leaf);
      Scalar *message = Branch(operation.branch);
      for (std::size_t parent_state = 0; parent_state < states_;
           ++parent_state) {
        Scalar value = 0.0;
        for (std::size_t child_state = 0; child_state < states_;
             ++child_state) {
          value +=
              path[parent_state * states_ + child_state] * leaf[child_state];
        }
        message[parent_state] = value;
      }
    }
  }

  void CombineBranches(std::span<const btrc::BranchCombination> operations) {
    for (const auto &operation : operations) {
      Scalar *left = Branch(operation.destination);
      const Scalar *right = Branch(operation.source);
      for (std::size_t state = 0; state < states_; ++state)
        left[state] *= right[state];
    }
  }

  void AbsorbBranches(std::span<const btrc::BranchAbsorption> operations) {
    for (const auto &operation : operations) {
      Scalar *node = Node(operation.parent);
      const Scalar *branch = Branch(operation.branch);
      for (std::size_t state = 0; state < states_; ++state)
        node[state] *= branch[state];
    }
  }

  void Compress(std::span<const btrc::Compression> operations) {
    for (const auto &operation : operations) {
      Scalar *left = Path(operation.left_edge);
      const Scalar *middle = Node(operation.middle);
      const Scalar *right = Path(operation.right_edge);
      std::copy(left, left + matrix_size_, storage_.reverse_scratch.begin());
      for (std::size_t parent_state = 0; parent_state < states_;
           ++parent_state) {
        for (std::size_t child_state = 0; child_state < states_;
             ++child_state) {
          Scalar value = 0.0;
          for (std::size_t middle_state = 0; middle_state < states_;
               ++middle_state) {
            value +=
                storage_
                    .reverse_scratch[parent_state * states_ + middle_state] *
                middle[middle_state] *
                right[middle_state * states_ + child_state];
          }
          left[parent_state * states_ + child_state] = value;
        }
      }
    }
  }

  Scalar FinishRoot() {
    const Scalar *root = Node(plan_.root());
    storage_.partition = std::accumulate(root, root + states_, Scalar{0});
    return storage_.partition;
  }

private:
  Scalar *Node(btrc::Index node) {
    return storage_.nodes.data() + node * states_;
  }
  const Scalar *Node(btrc::Index node) const {
    return storage_.nodes.data() + node * states_;
  }
  Scalar *Path(btrc::Index edge) {
    return storage_.paths.data() + edge * matrix_size_;
  }
  const Scalar *Path(btrc::Index edge) const {
    return storage_.paths.data() + edge * matrix_size_;
  }
  Scalar *Branch(btrc::Index branch) {
    return storage_.branches.data() + branch * states_;
  }
  const Scalar *Branch(btrc::Index branch) const {
    return storage_.branches.data() + branch * states_;
  }
  const btrc::Plan &plan_;
  std::size_t states_;
  std::size_t matrix_size_;
  Workspace::Impl &storage_;
};

class LogSumProductDispatcher {
public:
  LogSumProductDispatcher(const btrc::Plan &plan, std::size_t states,
                          Workspace::Impl &storage)
      : plan_(plan), states_(states), matrix_size_(states * states),
        storage_(storage) {
    std::transform(storage_.nodes.begin(), storage_.nodes.end(),
                   storage_.nodes.begin(), PotentialLog);
    std::transform(storage_.paths.begin(), storage_.paths.end(),
                   storage_.paths.begin(), PotentialLog);
  }

  void Rake(std::span<const btrc::Rake> operations) {
    for (const auto &operation : operations) {
      const Scalar *path = Path(operation.edge);
      const Scalar *leaf = Node(operation.leaf);
      std::copy(path, path + matrix_size_,
                storage_.rake_paths.begin() + operation.branch * matrix_size_);
      std::copy(leaf, leaf + states_,
                storage_.rake_leaves.begin() + operation.branch * states_);
      Scalar *message = Branch(operation.branch);
      for (std::size_t parent_state = 0; parent_state < states_;
           ++parent_state) {
        message[parent_state] =
            LogSumExp(states_, [&](std::size_t child_state) {
              return path[parent_state * states_ + child_state] +
                     leaf[child_state];
            });
      }
    }
  }

  void CombineBranches(std::span<const btrc::BranchCombination> operations) {
    for (const auto &operation : operations) {
      Scalar *destination = Branch(operation.destination);
      const Scalar *source = Branch(operation.source);
      for (std::size_t state = 0; state < states_; ++state)
        destination[state] += source[state];
    }
  }

  void AbsorbBranches(std::span<const btrc::BranchAbsorption> operations) {
    for (const auto &operation : operations) {
      Scalar *node = Node(operation.parent);
      const Scalar *branch = Branch(operation.branch);
      for (std::size_t state = 0; state < states_; ++state)
        node[state] += branch[state];
    }
  }

  void Compress(std::span<const btrc::Compression> operations) {
    for (const auto &operation : operations) {
      Scalar *left = Path(operation.left_edge);
      const Scalar *middle = Node(operation.middle);
      const Scalar *right = Path(operation.right_edge);
      Scalar *saved_left =
          storage_.compression_left.data() + operation.tape * matrix_size_;
      Scalar *saved_middle =
          storage_.compression_middle.data() + operation.tape * states_;
      Scalar *saved_right =
          storage_.compression_right.data() + operation.tape * matrix_size_;
      std::copy(left, left + matrix_size_, saved_left);
      std::copy(middle, middle + states_, saved_middle);
      std::copy(right, right + matrix_size_, saved_right);
      for (std::size_t parent_state = 0; parent_state < states_;
           ++parent_state) {
        for (std::size_t child_state = 0; child_state < states_;
             ++child_state) {
          left[parent_state * states_ + child_state] =
              LogSumExp(states_, [&](std::size_t middle_state) {
                return saved_left[parent_state * states_ + middle_state] +
                       saved_middle[middle_state] +
                       saved_right[middle_state * states_ + child_state];
              });
        }
      }
    }
  }

  Scalar FinishRoot() {
    const Scalar *root = Node(plan_.root());
    storage_.log_partition =
        LogSumExp(states_, [&](std::size_t state) { return root[state]; });
    if (!std::isfinite(storage_.log_partition)) {
      throw std::domain_error(
          "the tree HMM has a nonpositive partition function");
    }
    storage_.partition = std::exp(storage_.log_partition);
    return storage_.log_partition;
  }

  void SeedRootAdjoint() {
    const Scalar *root = Node(plan_.root());
    Scalar *root_adjoint = NodeAdjoint(plan_.root());
    for (std::size_t state = 0; state < states_; ++state)
      root_adjoint[state] = std::exp(root[state] - storage_.log_partition);
  }

  void SeedRootSample(std::span<const Scalar> uniforms) {
    if (uniforms.size() != plan_.num_nodes()) {
      throw std::invalid_argument(
          "posterior sampling requires one uniform variate per node");
    }
    for (Scalar uniform : uniforms) {
      if (!std::isfinite(uniform) || uniform < 0.0 || uniform >= 1.0) {
        throw std::invalid_argument(
            "posterior-sampling variates must lie in [0, 1)");
      }
    }
    uniforms_ = uniforms;
    const Scalar *root = Node(plan_.root());
    storage_.assignments[plan_.root()] = SampleLogWeights(
        states_, [&](std::size_t state) { return root[state]; },
        storage_.log_partition, uniforms[plan_.root()]);
  }

  void ExpandRakes(std::span<const btrc::Rake> operations) {
    for (const btrc::Rake &operation : operations) {
      const std::size_t parent_state = storage_.assignments[operation.parent];
      const Scalar *path =
          storage_.rake_paths.data() + operation.branch * matrix_size_;
      const Scalar *leaf =
          storage_.rake_leaves.data() + operation.branch * states_;
      Scalar *conditional = storage_.reverse_scratch.data();
      for (std::size_t child_state = 0; child_state < states_; ++child_state) {
        conditional[child_state] =
            path[parent_state * states_ + child_state] + leaf[child_state];
      }
      const Scalar normalizer =
          LogSumExp(states_, [&](std::size_t child_state) {
            return conditional[child_state];
          });
      storage_.assignments[operation.leaf] = SampleLogWeights(
          states_,
          [&](std::size_t child_state) { return conditional[child_state]; },
          normalizer, uniforms_[operation.leaf]);
    }
  }

  void ExpandCompressions(std::span<const btrc::Compression> operations) {
    for (const btrc::Compression &operation : operations) {
      const std::size_t parent_state = storage_.assignments[operation.parent];
      const std::size_t child_state = storage_.assignments[operation.child];
      const Scalar *left =
          storage_.compression_left.data() + operation.tape * matrix_size_;
      const Scalar *middle =
          storage_.compression_middle.data() + operation.tape * states_;
      const Scalar *right =
          storage_.compression_right.data() + operation.tape * matrix_size_;
      Scalar *conditional = storage_.reverse_scratch.data();
      for (std::size_t middle_state = 0; middle_state < states_;
           ++middle_state) {
        conditional[middle_state] =
            left[parent_state * states_ + middle_state] + middle[middle_state] +
            right[middle_state * states_ + child_state];
      }
      const Scalar normalizer =
          LogSumExp(states_, [&](std::size_t middle_state) {
            return conditional[middle_state];
          });
      storage_.assignments[operation.middle] = SampleLogWeights(
          states_,
          [&](std::size_t middle_state) { return conditional[middle_state]; },
          normalizer, uniforms_[operation.middle]);
    }
  }

  std::span<const std::size_t> Sample() const { return storage_.assignments; }

  void ReverseCompressions(std::span<const btrc::Compression> operations) {
    for (std::size_t index = operations.size(); index-- > 0;) {
      const auto &operation = operations[index];
      const Scalar *left =
          storage_.compression_left.data() + operation.tape * matrix_size_;
      const Scalar *middle =
          storage_.compression_middle.data() + operation.tape * states_;
      const Scalar *right =
          storage_.compression_right.data() + operation.tape * matrix_size_;
      Scalar *output_adjoint = PathAdjoint(operation.left_edge);
      std::fill(storage_.reverse_scratch.begin(),
                storage_.reverse_scratch.end(), 0.0);
      Scalar *middle_adjoint = NodeAdjoint(operation.middle);
      Scalar *right_adjoint = PathAdjoint(operation.right_edge);
      for (std::size_t parent_state = 0; parent_state < states_;
           ++parent_state) {
        for (std::size_t child_state = 0; child_state < states_;
             ++child_state) {
          const std::size_t output_index = parent_state * states_ + child_state;
          const Scalar adjoint = output_adjoint[output_index];
          const Scalar output =
              LogSumExp(states_, [&](std::size_t middle_state) {
                return left[parent_state * states_ + middle_state] +
                       middle[middle_state] +
                       right[middle_state * states_ + child_state];
              });
          if (adjoint == 0.0 || !std::isfinite(output))
            continue;
          for (std::size_t middle_state = 0; middle_state < states_;
               ++middle_state) {
            const std::size_t left_index =
                parent_state * states_ + middle_state;
            const std::size_t right_index =
                middle_state * states_ + child_state;
            const Scalar term =
                left[left_index] + middle[middle_state] + right[right_index];
            if (!std::isfinite(term))
              continue;
            const Scalar contribution = adjoint * std::exp(term - output);
            storage_.reverse_scratch[left_index] += contribution;
            middle_adjoint[middle_state] += contribution;
            right_adjoint[right_index] += contribution;
          }
        }
      }
      std::copy(storage_.reverse_scratch.begin(),
                storage_.reverse_scratch.end(), output_adjoint);
    }
  }

  void
  ReverseAbsorbBranches(std::span<const btrc::BranchAbsorption> operations) {
    for (std::size_t index = operations.size(); index-- > 0;) {
      const auto &operation = operations[index];
      const Scalar *node_adjoint = NodeAdjoint(operation.parent);
      Scalar *branch_adjoint = BranchAdjoint(operation.branch);
      for (std::size_t state = 0; state < states_; ++state)
        branch_adjoint[state] += node_adjoint[state];
    }
  }

  void
  ReverseCombineBranches(std::span<const btrc::BranchCombination> operations) {
    for (std::size_t index = operations.size(); index-- > 0;) {
      const auto &operation = operations[index];
      const Scalar *destination_adjoint = BranchAdjoint(operation.destination);
      Scalar *source_adjoint = BranchAdjoint(operation.source);
      for (std::size_t state = 0; state < states_; ++state)
        source_adjoint[state] += destination_adjoint[state];
    }
  }

  void ReverseRakes(std::span<const btrc::Rake> operations) {
    for (std::size_t index = operations.size(); index-- > 0;) {
      const auto &operation = operations[index];
      const Scalar *path =
          storage_.rake_paths.data() + operation.branch * matrix_size_;
      const Scalar *leaf =
          storage_.rake_leaves.data() + operation.branch * states_;
      const Scalar *message_adjoint = BranchAdjoint(operation.branch);
      Scalar *path_adjoint = PathAdjoint(operation.edge);
      Scalar *leaf_adjoint = NodeAdjoint(operation.leaf);
      for (std::size_t parent_state = 0; parent_state < states_;
           ++parent_state) {
        const Scalar message = LogSumExp(states_, [&](std::size_t child_state) {
          return path[parent_state * states_ + child_state] + leaf[child_state];
        });
        if (!std::isfinite(message))
          continue;
        for (std::size_t child_state = 0; child_state < states_;
             ++child_state) {
          const std::size_t path_index = parent_state * states_ + child_state;
          const Scalar term = path[path_index] + leaf[child_state];
          if (!std::isfinite(term))
            continue;
          const Scalar contribution =
              message_adjoint[parent_state] * std::exp(term - message);
          path_adjoint[path_index] += contribution;
          leaf_adjoint[child_state] += contribution;
        }
      }
    }
  }

  MarginalView BuildMarginals() {
    return {storage_.partition, storage_.log_partition, storage_.node_adjoints,
            storage_.path_adjoints};
  }

private:
  Scalar *Node(btrc::Index node) {
    return storage_.nodes.data() + node * states_;
  }
  const Scalar *Node(btrc::Index node) const {
    return storage_.nodes.data() + node * states_;
  }
  Scalar *Path(btrc::Index edge) {
    return storage_.paths.data() + edge * matrix_size_;
  }
  const Scalar *Path(btrc::Index edge) const {
    return storage_.paths.data() + edge * matrix_size_;
  }
  Scalar *Branch(btrc::Index branch) {
    return storage_.branches.data() + branch * states_;
  }
  Scalar *NodeAdjoint(btrc::Index node) {
    return storage_.node_adjoints.data() + node * states_;
  }
  const Scalar *NodeAdjoint(btrc::Index node) const {
    return storage_.node_adjoints.data() + node * states_;
  }
  Scalar *PathAdjoint(btrc::Index edge) {
    return storage_.path_adjoints.data() + edge * matrix_size_;
  }
  Scalar *BranchAdjoint(btrc::Index branch) {
    return storage_.branch_adjoints.data() + branch * states_;
  }
  const Scalar *BranchAdjoint(btrc::Index branch) const {
    return storage_.branch_adjoints.data() + branch * states_;
  }

  const btrc::Plan &plan_;
  std::size_t states_;
  std::size_t matrix_size_;
  Workspace::Impl &storage_;
  std::span<const Scalar> uniforms_;
};

class MaxProductDispatcher {
public:
  MaxProductDispatcher(const btrc::Plan &plan, std::size_t states,
                       Workspace::Impl &storage)
      : plan_(plan), states_(states), matrix_size_(states * states),
        storage_(storage) {
    std::transform(storage_.nodes.begin(), storage_.nodes.end(),
                   storage_.nodes.begin(), PotentialLog);
    std::transform(storage_.paths.begin(), storage_.paths.end(),
                   storage_.paths.begin(), PotentialLog);
  }

  void Rake(std::span<const btrc::Rake> operations) {
    for (const btrc::Rake &operation : operations) {
      const Scalar *path = Path(operation.edge);
      const Scalar *leaf = Node(operation.leaf);
      Scalar *message = Branch(operation.branch);
      std::size_t *choices =
          storage_.rake_choices.data() + operation.branch * states_;
      for (std::size_t parent_state = 0; parent_state < states_;
           ++parent_state) {
        choices[parent_state] = Argmax(
            states_,
            [&](std::size_t child_state) {
              return path[parent_state * states_ + child_state] +
                     leaf[child_state];
            },
            &message[parent_state]);
      }
    }
  }

  void CombineBranches(std::span<const btrc::BranchCombination> operations) {
    for (const btrc::BranchCombination &operation : operations) {
      Scalar *destination = Branch(operation.destination);
      const Scalar *source = Branch(operation.source);
      for (std::size_t state = 0; state < states_; ++state)
        destination[state] += source[state];
    }
  }

  void AbsorbBranches(std::span<const btrc::BranchAbsorption> operations) {
    for (const btrc::BranchAbsorption &operation : operations) {
      Scalar *node = Node(operation.parent);
      const Scalar *branch = Branch(operation.branch);
      for (std::size_t state = 0; state < states_; ++state)
        node[state] += branch[state];
    }
  }

  void Compress(std::span<const btrc::Compression> operations) {
    for (const btrc::Compression &operation : operations) {
      Scalar *left = Path(operation.left_edge);
      const Scalar *middle = Node(operation.middle);
      const Scalar *right = Path(operation.right_edge);
      std::copy(left, left + matrix_size_, storage_.reverse_scratch.begin());
      std::size_t *choices =
          storage_.compression_choices.data() + operation.tape * matrix_size_;
      for (std::size_t parent_state = 0; parent_state < states_;
           ++parent_state) {
        for (std::size_t child_state = 0; child_state < states_;
             ++child_state) {
          const std::size_t output = parent_state * states_ + child_state;
          choices[output] = Argmax(
              states_,
              [&](std::size_t middle_state) {
                return storage_.reverse_scratch[parent_state * states_ +
                                                middle_state] +
                       middle[middle_state] +
                       right[middle_state * states_ + child_state];
              },
              &left[output]);
        }
      }
    }
  }

  void FinishRoot() {
    const Scalar *root = Node(plan_.root());
    storage_.assignments[plan_.root()] = Argmax(
        states_, [&](std::size_t state) { return root[state]; },
        &storage_.maximum_log_weight);
    if (!std::isfinite(storage_.maximum_log_weight)) {
      throw std::domain_error("the tree HMM has no positive-weight assignment");
    }
  }

  void ExpandRakes(std::span<const btrc::Rake> operations) {
    for (const btrc::Rake &operation : operations) {
      const std::size_t parent_state = storage_.assignments[operation.parent];
      storage_.assignments[operation.leaf] =
          storage_.rake_choices[operation.branch * states_ + parent_state];
    }
  }

  void ExpandCompressions(std::span<const btrc::Compression> operations) {
    for (const btrc::Compression &operation : operations) {
      const std::size_t parent_state = storage_.assignments[operation.parent];
      const std::size_t child_state = storage_.assignments[operation.child];
      storage_.assignments[operation.middle] =
          storage_.compression_choices[operation.tape * matrix_size_ +
                                       parent_state * states_ + child_state];
    }
  }

  MaximumAssignmentView Result() const {
    return {std::exp(storage_.maximum_log_weight), storage_.maximum_log_weight,
            storage_.assignments};
  }

private:
  Scalar *Node(btrc::Index node) {
    return storage_.nodes.data() + node * states_;
  }
  const Scalar *Node(btrc::Index node) const {
    return storage_.nodes.data() + node * states_;
  }
  Scalar *Path(btrc::Index edge) {
    return storage_.paths.data() + edge * matrix_size_;
  }
  const Scalar *Path(btrc::Index edge) const {
    return storage_.paths.data() + edge * matrix_size_;
  }
  Scalar *Branch(btrc::Index branch) {
    return storage_.branches.data() + branch * states_;
  }

  const btrc::Plan &plan_;
  std::size_t states_;
  std::size_t matrix_size_;
  Workspace::Impl &storage_;
};

class ScaledSumProductDispatcher {
public:
  ScaledSumProductDispatcher(const btrc::Plan &plan, std::size_t states,
                             Workspace::Impl &storage)
      : plan_(plan), states_(states), matrix_size_(states * states),
        storage_(storage) {}

  void Rake(std::span<const btrc::Rake> operations) {
    for (const auto &operation : operations) {
      const Scalar *path = Path(operation.edge);
      const Scalar *leaf = Node(operation.leaf);
      Scalar *message = Branch(operation.branch);
      for (std::size_t parent_state = 0; parent_state < states_;
           ++parent_state) {
        Scalar value = 0.0;
        for (std::size_t child_state = 0; child_state < states_;
             ++child_state) {
          value +=
              path[parent_state * states_ + child_state] * leaf[child_state];
        }
        message[parent_state] = value;
      }
      BranchScale(operation.branch) =
          Normalize(message, states_,
                    PathScale(operation.edge) + NodeScale(operation.leaf));
    }
  }

  void CombineBranches(std::span<const btrc::BranchCombination> operations) {
    for (const auto &operation : operations) {
      Scalar *destination = Branch(operation.destination);
      const Scalar *source = Branch(operation.source);
      for (std::size_t state = 0; state < states_; ++state)
        destination[state] *= source[state];
      BranchScale(operation.destination) = Normalize(
          destination, states_,
          BranchScale(operation.destination) + BranchScale(operation.source));
    }
  }

  void AbsorbBranches(std::span<const btrc::BranchAbsorption> operations) {
    for (const auto &operation : operations) {
      Scalar *node = Node(operation.parent);
      const Scalar *branch = Branch(operation.branch);
      for (std::size_t state = 0; state < states_; ++state)
        node[state] *= branch[state];
      NodeScale(operation.parent) = Normalize(
          node, states_,
          NodeScale(operation.parent) + BranchScale(operation.branch));
    }
  }

  void Compress(std::span<const btrc::Compression> operations) {
    for (const auto &operation : operations) {
      Scalar *left = Path(operation.left_edge);
      const Scalar *middle = Node(operation.middle);
      const Scalar *right = Path(operation.right_edge);
      std::copy(left, left + matrix_size_, storage_.reverse_scratch.begin());
      for (std::size_t parent_state = 0; parent_state < states_;
           ++parent_state) {
        for (std::size_t child_state = 0; child_state < states_;
             ++child_state) {
          Scalar value = 0.0;
          for (std::size_t middle_state = 0; middle_state < states_;
               ++middle_state) {
            value +=
                storage_
                    .reverse_scratch[parent_state * states_ + middle_state] *
                middle[middle_state] *
                right[middle_state * states_ + child_state];
          }
          left[parent_state * states_ + child_state] = value;
        }
      }
      PathScale(operation.left_edge) = Normalize(
          left, matrix_size_,
          PathScale(operation.left_edge) + NodeScale(operation.middle) +
              PathScale(operation.right_edge));
    }
  }

  Scalar FinishRoot() const {
    const Scalar *root = Node(plan_.root());
    const Scalar sum = std::accumulate(root, root + states_, Scalar{0});
    if (!(sum > 0.0))
      return -std::numeric_limits<Scalar>::infinity();
    return NodeScale(plan_.root()) + std::log(sum);
  }

private:
  static Scalar Normalize(Scalar *values, std::size_t size,
                          Scalar input_scale) {
    const Scalar maximum = *std::max_element(values, values + size);
    if (!(maximum > 0.0)) {
      return -std::numeric_limits<Scalar>::infinity();
    }
    const Scalar inverse = Scalar{1} / maximum;
    for (std::size_t index = 0; index < size; ++index)
      values[index] *= inverse;
    return input_scale + std::log(maximum);
  }

  Scalar *Node(btrc::Index node) {
    return storage_.nodes.data() + node * states_;
  }
  const Scalar *Node(btrc::Index node) const {
    return storage_.nodes.data() + node * states_;
  }
  Scalar *Path(btrc::Index edge) {
    return storage_.paths.data() + edge * matrix_size_;
  }
  const Scalar *Path(btrc::Index edge) const {
    return storage_.paths.data() + edge * matrix_size_;
  }
  Scalar *Branch(btrc::Index branch) {
    return storage_.branches.data() + branch * states_;
  }
  const Scalar *Branch(btrc::Index branch) const {
    return storage_.branches.data() + branch * states_;
  }
  Scalar &NodeScale(btrc::Index node) { return storage_.node_log_scales[node]; }
  Scalar NodeScale(btrc::Index node) const {
    return storage_.node_log_scales[node];
  }
  Scalar &PathScale(btrc::Index edge) { return storage_.path_log_scales[edge]; }
  Scalar PathScale(btrc::Index edge) const {
    return storage_.path_log_scales[edge];
  }
  Scalar &BranchScale(btrc::Index branch) {
    return storage_.branch_log_scales[branch];
  }
  Scalar BranchScale(btrc::Index branch) const {
    return storage_.branch_log_scales[branch];
  }

  const btrc::Plan &plan_;
  std::size_t states_;
  std::size_t matrix_size_;
  Workspace::Impl &storage_;
};

} // namespace

Workspace::Workspace() : impl_(std::make_unique<Impl>()) {}
Workspace::~Workspace() = default;
Workspace::Workspace(Workspace &&) noexcept = default;
Workspace &Workspace::operator=(Workspace &&) noexcept = default;

void Workspace::Reserve(const btrc::Plan &plan, std::size_t states) {
  if (states == 0)
    throw std::invalid_argument("a tree HMM must have at least one state");
  const std::size_t matrix_size =
      CheckedProduct(states, states, "state matrix size");
  Impl &storage = *impl_;
  storage.plan = &plan;
  storage.states = states;
  const std::size_t node_values =
      CheckedProduct(plan.num_nodes(), states, "node workspace");
  const std::size_t path_values =
      CheckedProduct(plan.num_edges(), matrix_size, "path workspace");
  const std::size_t branch_values =
      CheckedProduct(plan.num_branches(), states, "branch workspace");
  const std::size_t compression_matrices = CheckedProduct(
      plan.num_compressions(), matrix_size, "compression matrix tape");
  const std::size_t compression_vectors = CheckedProduct(
      plan.num_compressions(), states, "compression vector tape");

  storage.nodes.resize(node_values);
  storage.paths.resize(path_values);
  storage.branches.resize(branch_values);
  storage.node_log_scales.resize(plan.num_nodes());
  storage.path_log_scales.resize(plan.num_edges());
  storage.branch_log_scales.resize(plan.num_branches());
  storage.node_adjoints.resize(node_values);
  storage.path_adjoints.resize(path_values);
  storage.branch_adjoints.resize(branch_values);
  storage.rake_paths.resize(
      CheckedProduct(plan.num_branches(), matrix_size, "rake matrix tape"));
  storage.rake_leaves.resize(branch_values);
  storage.compression_left.resize(compression_matrices);
  storage.compression_middle.resize(compression_vectors);
  storage.compression_right.resize(compression_matrices);
  storage.reverse_scratch.resize(matrix_size);
  storage.rake_choices.resize(branch_values);
  storage.compression_choices.resize(compression_matrices);
  storage.assignments.resize(plan.num_nodes());
}

Scalar PartitionFunctionPrepared(ModelView model, Workspace &workspace) {
  Workspace::Impl &storage = Prepare(model, *workspace.impl_);
  SumProductDispatcher dispatcher(model.plan, model.states, storage);
  btrc::Contract(model.plan, dispatcher);
  return dispatcher.FinishRoot();
}

Scalar LogPartitionFunctionPrepared(ModelView model, Workspace &workspace) {
  Workspace::Impl &storage = Prepare(model, *workspace.impl_);
  std::fill(storage.node_log_scales.begin(), storage.node_log_scales.end(),
            0.0);
  std::fill(storage.path_log_scales.begin(), storage.path_log_scales.end(),
            0.0);
  std::fill(storage.branch_log_scales.begin(), storage.branch_log_scales.end(),
            0.0);
  ScaledSumProductDispatcher dispatcher(model.plan, model.states, storage);
  btrc::Contract(model.plan, dispatcher);
  return dispatcher.FinishRoot();
}

MarginalView PosteriorMarginalsPrepared(ModelView model, Workspace &workspace) {
  Workspace::Impl &storage = Prepare(model, *workspace.impl_);
  std::fill(storage.node_adjoints.begin(), storage.node_adjoints.end(), 0.0);
  std::fill(storage.path_adjoints.begin(), storage.path_adjoints.end(), 0.0);
  std::fill(storage.branch_adjoints.begin(), storage.branch_adjoints.end(),
            0.0);
  LogSumProductDispatcher dispatcher(model.plan, model.states, storage);
  btrc::Contract(model.plan, dispatcher);
  dispatcher.FinishRoot();
  dispatcher.SeedRootAdjoint();
  btrc::Reverse(model.plan, dispatcher);
  return dispatcher.BuildMarginals();
}

MaximumAssignmentView MaximumAPosterioriPrepared(ModelView model,
                                                 Workspace &workspace) {
  Workspace::Impl &storage = Prepare(model, *workspace.impl_);
  MaxProductDispatcher dispatcher(model.plan, model.states, storage);
  btrc::Contract(model.plan, dispatcher);
  dispatcher.FinishRoot();
  btrc::Expand(model.plan, dispatcher);
  return dispatcher.Result();
}

std::span<const std::size_t>
PosteriorSamplePrepared(ModelView model, std::span<const Scalar> uniforms,
                        Workspace &workspace) {
  Workspace::Impl &storage = Prepare(model, *workspace.impl_);
  LogSumProductDispatcher dispatcher(model.plan, model.states, storage);
  btrc::Contract(model.plan, dispatcher);
  dispatcher.FinishRoot();
  dispatcher.SeedRootSample(uniforms);
  btrc::Expand(model.plan, dispatcher);
  return dispatcher.Sample();
}

Marginals Materialize(MarginalView view) {
  return {
      .partition = view.partition,
      .log_partition = view.log_partition,
      .nodes = std::vector<Scalar>(view.nodes.begin(), view.nodes.end()),
      .edges = std::vector<Scalar>(view.edges.begin(), view.edges.end()),
  };
}

MaximumAssignment Materialize(MaximumAssignmentView view) {
  return {
      .weight = view.weight,
      .log_weight = view.log_weight,
      .states =
          std::vector<std::size_t>(view.states.begin(), view.states.end()),
  };
}

Scalar PartitionFunction(ModelView model) {
  Workspace workspace;
  workspace.Reserve(model.plan, model.states);
  return PartitionFunctionPrepared(model, workspace);
}

Scalar LogPartitionFunction(ModelView model) {
  Workspace workspace;
  workspace.Reserve(model.plan, model.states);
  return LogPartitionFunctionPrepared(model, workspace);
}

Marginals PosteriorMarginals(ModelView model) {
  Workspace workspace;
  workspace.Reserve(model.plan, model.states);
  return Materialize(PosteriorMarginalsPrepared(model, workspace));
}

MaximumAssignment MaximumAPosteriori(ModelView model) {
  Workspace workspace;
  workspace.Reserve(model.plan, model.states);
  return Materialize(MaximumAPosterioriPrepared(model, workspace));
}

std::vector<std::size_t> PosteriorSample(ModelView model,
                                         std::span<const Scalar> uniforms) {
  Workspace workspace;
  workspace.Reserve(model.plan, model.states);
  const std::span<const std::size_t> sample =
      PosteriorSamplePrepared(model, uniforms, workspace);
  return {sample.begin(), sample.end()};
}

} // namespace tree_hmm
