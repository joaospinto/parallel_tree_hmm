#ifndef TREE_HMM_SRC_ACCELERATOR_PATH_STORAGE_H_
#define TREE_HMM_SRC_ACCELERATOR_PATH_STORAGE_H_

#include "btrc/plan.h"

#include <cstddef>
#include <limits>
#include <stdexcept>
#include <vector>

namespace tree_hmm::internal {

inline constexpr btrc::Index kImmutablePath =
    std::numeric_limits<btrc::Index>::max();

struct AcceleratorPathStorage {
  std::vector<btrc::Index> mutable_slot_by_edge;
  std::vector<btrc::Index> edge_by_mutable_slot;
};

inline std::size_t AcceleratorPathCount(std::size_t edges,
                                        std::size_t mutable_paths,
                                        std::size_t batch) {
  if (mutable_paths != 0 &&
      batch >
          (std::numeric_limits<std::size_t>::max() - edges) / mutable_paths) {
    throw std::length_error("accelerator path count overflows size_t");
  }
  return edges + mutable_paths * batch;
}

inline AcceleratorPathStorage
MakeAcceleratorPathStorage(const btrc::Plan &plan) {
  AcceleratorPathStorage result;
  result.mutable_slot_by_edge.assign(plan.num_edges(), kImmutablePath);
  for (const btrc::Compression &compression : plan.compressions())
    result.mutable_slot_by_edge[compression.left_edge] = 0;
  for (std::size_t edge = 0; edge < plan.num_edges(); ++edge) {
    if (result.mutable_slot_by_edge[edge] == kImmutablePath)
      continue;
    result.mutable_slot_by_edge[edge] =
        static_cast<btrc::Index>(result.edge_by_mutable_slot.size());
    result.edge_by_mutable_slot.push_back(static_cast<btrc::Index>(edge));
  }
  return result;
}

} // namespace tree_hmm::internal

#endif // TREE_HMM_SRC_ACCELERATOR_PATH_STORAGE_H_
