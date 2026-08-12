#ifndef TREE_HMM_INFERENCE_H_
#define TREE_HMM_INFERENCE_H_

#include <cstddef>
#include <memory>
#include <span>
#include <vector>

#include "btrc/plan.h"

namespace tree_hmm {

// Dense nonnegative factorization of a hidden Markov tree. Node potentials
// are [node, state]. Edge potentials are [edge, parent state, child state],
// using the edge order exposed by btrc::Plan.
struct ModelView {
  const btrc::Plan &plan;
  std::size_t states;
  std::span<const double> node_potentials;
  std::span<const double> edge_potentials;
};

struct Marginals {
  double partition = 0.0;
  double log_partition = 0.0;
  std::vector<double> nodes;
  std::vector<double> edges;
};

struct MarginalView {
  double partition = 0.0;
  double log_partition = 0.0;
  std::span<const double> nodes;
  std::span<const double> edges;
};

class Workspace {
public:
  struct Impl;

  Workspace();
  ~Workspace();
  Workspace(Workspace &&) noexcept;
  Workspace &operator=(Workspace &&) noexcept;
  Workspace(const Workspace &) = delete;
  Workspace &operator=(const Workspace &) = delete;

  // All host storage needed by repeated inference is allocated here.
  void Reserve(const btrc::Plan &plan, std::size_t states);

private:
  friend double PartitionFunctionPrepared(ModelView, Workspace &);
  friend double LogPartitionFunctionPrepared(ModelView, Workspace &);
  friend MarginalView PosteriorMarginalsPrepared(ModelView, Workspace &);
  std::unique_ptr<Impl> impl_;
};

double PartitionFunctionPrepared(ModelView model, Workspace &workspace);
double LogPartitionFunctionPrepared(ModelView model, Workspace &workspace);
MarginalView PosteriorMarginalsPrepared(ModelView model, Workspace &workspace);
Marginals Materialize(MarginalView view);

double PartitionFunction(ModelView model);
double LogPartitionFunction(ModelView model);

// Returns normalized p(x_i) and p(x_parent, x_child). Log-domain contraction
// and analytic reverse contraction avoid underflow and division by
// intermediate messages, including messages containing zeros.
Marginals PosteriorMarginals(ModelView model);

} // namespace tree_hmm

#endif // TREE_HMM_INFERENCE_H_
