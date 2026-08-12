#ifndef TREE_HMM_ACCELERATOR_H_
#define TREE_HMM_ACCELERATOR_H_

#include <cstddef>
#include <cstdint>
#include <span>

#include "btrc/plan.h"

namespace tree_hmm {

// A batch shares one topology and one set of edge transition factors. Node
// factors vary across the batch, as they do for independent alignment sites.
struct BatchedModelView {
  const btrc::Plan &plan;
  std::size_t states;
  std::size_t batch;
  // [batch, node, state]
  std::span<const float> node_potentials;
  // [edge, parent state, child state], broadcast across the batch.
  std::span<const float> edge_potentials;
};

// Host-writable storage with the same layout as BatchedModelView. Accelerator
// workspaces expose this view so applications can prepare factors directly in
// reusable pinned or shared storage instead of allocating or staging another
// full batch. Conversion preserves the ordinary inference API.
struct MutableBatchedModelView {
  const btrc::Plan &plan;
  std::size_t states;
  std::size_t batch;
  std::span<float> node_potentials;
  std::span<float> edge_potentials;

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
  std::span<const float> root_potential;
  // [category, state]
  std::span<const float> emission_potentials;
  // [edge, parent state, child state], broadcast across the batch.
  std::span<const float> edge_potentials;
};

struct MutableBatchedCategoricalModelView {
  const btrc::Plan &plan;
  std::size_t states;
  std::size_t batch;
  std::size_t categories;
  std::span<const btrc::Index> observation_nodes;
  std::span<std::uint8_t> observations;
  std::span<float> root_potential;
  std::span<float> emission_potentials;
  std::span<float> edge_potentials;

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

struct AcceleratorTimings {
  double upload_ms = 0.0;
  double kernel_ms = 0.0;
  double download_ms = 0.0;
  double wall_ms = 0.0;
};

// The returned values are owned by the workspace and remain valid until that
// workspace is reserved again, reused, moved, or destroyed.
struct PartitionView {
  std::span<const float> values;
  AcceleratorTimings timings;
};

} // namespace tree_hmm

#endif // TREE_HMM_ACCELERATOR_H_
