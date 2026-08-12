#ifndef TREE_HMM_ACCELERATOR_H_
#define TREE_HMM_ACCELERATOR_H_

#include <cstddef>
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
