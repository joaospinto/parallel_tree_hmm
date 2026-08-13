#include "src/accelerator_path_storage.h"

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace {

void TestLayout(const std::vector<std::int64_t> &parents) {
  const btrc::Plan plan = btrc::MakePlan(parents);
  const tree_hmm::internal::AcceleratorPathStorage storage =
      tree_hmm::internal::MakeAcceleratorPathStorage(plan);
  if (storage.mutable_slot_by_edge.size() != plan.num_edges())
    throw std::runtime_error("path layout has the wrong edge count");

  std::vector<bool> written(plan.num_edges());
  for (const btrc::Compression &compression : plan.compressions())
    written[compression.left_edge] = true;

  std::size_t expected_mutable_paths = 0;
  for (std::size_t edge = 0; edge < plan.num_edges(); ++edge) {
    const btrc::Index slot = storage.mutable_slot_by_edge[edge];
    if (!written[edge]) {
      if (slot != tree_hmm::internal::kImmutablePath)
        throw std::runtime_error("immutable edge has a mutable path slot");
      continue;
    }
    if (slot != expected_mutable_paths ||
        storage.edge_by_mutable_slot[slot] != edge) {
      throw std::runtime_error("mutable path maps are not inverse");
    }
    ++expected_mutable_paths;
  }
  if (storage.edge_by_mutable_slot.size() != expected_mutable_paths)
    throw std::runtime_error("path layout has the wrong mutable-path count");
}

} // namespace

int main() {
  TestLayout({-1, 0, 0, 1, 1, 2, 2});
  TestLayout({-1, 0, 1, 2, 3, 4, 5, 6, 7});
  TestLayout({-1, 0, 0, 0, 0, 0, 0, 0});

  constexpr std::size_t kEdges = 37;
  constexpr std::size_t kMutablePaths = 11;
  constexpr std::size_t kBatch = 19;
  if (tree_hmm::internal::AcceleratorPathCount(kEdges, kMutablePaths, kBatch) !=
      kEdges + kMutablePaths * kBatch) {
    throw std::runtime_error("accelerator path count is wrong");
  }
}
