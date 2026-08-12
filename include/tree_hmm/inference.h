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

struct MaximumAssignment {
  double weight = 0.0;
  double log_weight = 0.0;
  std::vector<std::size_t> states;
};

struct MaximumAssignmentView {
  double weight = 0.0;
  double log_weight = 0.0;
  std::span<const std::size_t> states;
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
  friend MaximumAssignmentView MaximumAPosterioriPrepared(ModelView,
                                                          Workspace &);
  friend std::span<const std::size_t>
  PosteriorSamplePrepared(ModelView, std::span<const double>, Workspace &);
  std::unique_ptr<Impl> impl_;
};

double PartitionFunctionPrepared(ModelView model, Workspace &workspace);
double LogPartitionFunctionPrepared(ModelView model, Workspace &workspace);
MarginalView PosteriorMarginalsPrepared(ModelView model, Workspace &workspace);
Marginals Materialize(MarginalView view);
MaximumAssignmentView MaximumAPosterioriPrepared(ModelView model,
                                                 Workspace &workspace);
MaximumAssignment Materialize(MaximumAssignmentView view);

// uniforms contains one independent U[0,1) variate per original node. Keeping
// random-number generation outside the inference algebra makes sampling
// reproducible and avoids allocating random-number-generator state within
// CPU or accelerator inference.
std::span<const std::size_t>
PosteriorSamplePrepared(ModelView model, std::span<const double> uniforms,
                        Workspace &workspace);

double PartitionFunction(ModelView model);
double LogPartitionFunction(ModelView model);

// Returns normalized p(x_i) and p(x_parent, x_child). Log-domain contraction
// and analytic reverse contraction avoid underflow and division by
// intermediate messages, including messages containing zeros.
Marginals PosteriorMarginals(ModelView model);
MaximumAssignment MaximumAPosteriori(ModelView model);
std::vector<std::size_t> PosteriorSample(ModelView model,
                                         std::span<const double> uniforms);

} // namespace tree_hmm

#endif // TREE_HMM_INFERENCE_H_
