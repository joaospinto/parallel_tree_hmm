#ifndef TREE_HMM_ACCELERATOR_H_
#define TREE_HMM_ACCELERATOR_H_

#include <cstddef>
#include <cstdint>
#include <span>

#include "btrc/plan.h"
#include "tree_hmm/scalar.h"

namespace tree_hmm {

// A batch shares one topology and one set of edge transition factors. Node
// factors vary across the batch, as they do for independent alignment sites.
struct BatchedModelView {
  const btrc::Plan &plan;
  std::size_t states;
  std::size_t batch;
  // [batch, node, state]
  std::span<const Scalar> node_potentials;
  // [edge, parent state, child state], broadcast across the batch.
  std::span<const Scalar> edge_potentials;
};

// Host-writable storage with the same layout as BatchedModelView. Accelerator
// workspaces expose this view so applications can prepare factors directly in
// reusable pinned or shared storage instead of allocating or staging another
// full batch. Conversion preserves the ordinary inference API.
struct MutableBatchedModelView {
  const btrc::Plan &plan;
  std::size_t states;
  std::size_t batch;
  std::span<Scalar> node_potentials;
  std::span<Scalar> edge_potentials;

  operator BatchedModelView() const {
    return {plan, states, batch, node_potentials, edge_potentials};
  }
};

// A batch of hidden Markov trees whose selected nodes carry categorical
// observations. Each observation is a byte indexing one row of the shared
// [category, state] emission table. This avoids materializing dense node
// potentials when most nodes are unobserved.
struct BatchedCategoricalModelView {
  const btrc::Plan &plan;
  std::size_t states;
  std::size_t batch;
  std::size_t categories;
  // Strictly increasing node indices.
  std::span<const btrc::Index> observation_nodes;
  // [batch, observation node]
  std::span<const std::uint8_t> observations;
  // [state]
  std::span<const Scalar> root_potential;
  // [category, state]
  std::span<const Scalar> emission_potentials;
  // [edge, parent state, child state], broadcast across the batch.
  std::span<const Scalar> edge_potentials;
};

struct MutableBatchedCategoricalModelView {
  const btrc::Plan &plan;
  std::size_t states;
  std::size_t batch;
  std::size_t categories;
  std::span<const btrc::Index> observation_nodes;
  std::span<std::uint8_t> observations;
  std::span<Scalar> root_potential;
  std::span<Scalar> emission_potentials;
  std::span<Scalar> edge_potentials;

  operator BatchedCategoricalModelView() const {
    return {plan,
            states,
            batch,
            categories,
            observation_nodes,
            observations,
            root_potential,
            emission_potentials,
            edge_potentials};
  }
};

// Selects which categorical inputs are staged before a prepared accelerator
// call. kFactors and kNone reuse observations staged by the latest kAll call
// for the same batch size; kNone also reuses the numerical factors staged by
// that call (or a later kFactors call). ReserveCategorical invalidates all
// staged inputs. A caller must not modify workspace-backed storage for a
// component it asks the backend to reuse.
enum class CategoricalInputUpdate {
  kAll,
  kFactors,
  kNone,
};

struct AcceleratorTimings {
  // Host-to-device transfers requested by CategoricalInputUpdate (or all
  // inputs for a dense model). Metal reports host writes to shared buffers.
  double upload_ms = 0.0;
  // Accelerator command execution, excluding the transfers above and result
  // transfer below.
  double kernel_ms = 0.0;
  // Device-to-host result transfers. Shared-memory Metal results require no
  // explicit transfer and report zero.
  double download_ms = 0.0;
  // Entire prepared-call latency, including validation, staging, command
  // submission, synchronization, and output checks.
  double wall_ms = 0.0;
};

// The returned values are owned by the workspace and remain valid until that
// workspace is reserved again, reused, moved, or destroyed.
struct PartitionView {
  std::span<const Scalar> values;
  AcceleratorTimings timings;
};

// One joint maximum-weight assignment per batch item. Log weights have shape
// [batch], and states have shape [batch, node]. The spans are owned by the
// backend workspace and follow the same lifetime rules as PartitionView.
struct BatchedMaximumAssignmentView {
  std::span<const Scalar> log_weights;
  std::span<const std::uint32_t> states;
  AcceleratorTimings timings;
};

// One posterior draw per batch item. States have shape [batch, node] and are
// owned by the backend workspace.
struct BatchedPosteriorSampleView {
  std::span<const std::uint32_t> states;
  AcceleratorTimings timings;
};

// Posterior probabilities for every batch item. Log partitions have shape
// [batch], nodes have shape [batch, node, state], and edges have shape
// [batch, edge, parent state, child state]. All spans are owned by the backend
// workspace.
struct BatchedMarginalView {
  std::span<const Scalar> log_partitions;
  std::span<const Scalar> nodes;
  std::span<const Scalar> edges;
  AcceleratorTimings timings;
};

} // namespace tree_hmm

#endif // TREE_HMM_ACCELERATOR_H_
