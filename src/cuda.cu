#include "tree_hmm/cuda.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "src/cuda_device_algebra.h"

namespace tree_hmm::cuda {
namespace {

using Clock = std::chrono::steady_clock;

struct Params {
  std::uint32_t states;
  std::uint32_t nodes;
  std::uint32_t edges;
  std::uint32_t branches;
  std::uint32_t batch;
  std::uint32_t root;
  std::uint32_t operation_offset;
  std::uint32_t operation_count;
  std::uint32_t scaled;
  std::uint32_t paths_batched;
};

static_assert(sizeof(btrc::Rake) == 16);
static_assert(sizeof(btrc::BranchCombination) == 12);
static_assert(sizeof(btrc::BranchAbsorption) == 12);
static_assert(sizeof(btrc::Compression) == 24);
static_assert(sizeof(Params) == 40);

void Check(cudaError_t status, const char *operation) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(operation) + ": " +
                             cudaGetErrorString(status));
  }
}

std::uint32_t CheckedU32(std::size_t value, const char *description) {
  if (value > std::numeric_limits<std::uint32_t>::max())
    throw std::length_error(std::string(description) + " exceeds uint32_t");
  return static_cast<std::uint32_t>(value);
}

std::size_t CheckedProduct(std::initializer_list<std::size_t> values,
                           const char *description) {
  std::size_t result = 1;
  for (const std::size_t value : values) {
    if (value != 0 && result > std::numeric_limits<std::size_t>::max() / value)
      throw std::length_error(std::string(description) + " overflows size_t");
    result *= value;
  }
  return result;
}

template <class Value> void DeviceAllocate(Value *&pointer, std::size_t count) {
  Check(
      cudaMalloc(reinterpret_cast<void **>(&pointer),
                 std::max<std::size_t>(
                     CheckedProduct({count, sizeof(Value)}, "CUDA buffer"), 1)),
      "cudaMalloc");
}

template <class Value> void HostAllocate(Value *&pointer, std::size_t count) {
  Check(
      cudaMallocHost(
          reinterpret_cast<void **>(&pointer),
          std::max<std::size_t>(
              CheckedProduct({count, sizeof(Value)}, "pinned host buffer"), 1)),
      "cudaMallocHost");
}

template <class Value> void DeviceFree(Value *&pointer) noexcept {
  if (pointer != nullptr) {
    static_cast<void>(cudaFree(pointer));
    pointer = nullptr;
  }
}

template <class Value> void HostFree(Value *&pointer) noexcept {
  if (pointer != nullptr) {
    static_cast<void>(cudaFreeHost(pointer));
    pointer = nullptr;
  }
}

void Destroy(cudaEvent_t &event) noexcept {
  if (event != nullptr) {
    static_cast<void>(cudaEventDestroy(event));
    event = nullptr;
  }
}

void Destroy(cudaStream_t &stream) noexcept {
  if (stream != nullptr) {
    static_cast<void>(cudaStreamDestroy(stream));
    stream = nullptr;
  }
}

template <class Value>
void Upload(Value *destination, std::span<const Value> source,
            cudaStream_t stream) {
  if (!source.empty()) {
    Check(cudaMemcpyAsync(destination, source.data(), source.size_bytes(),
                          cudaMemcpyHostToDevice, stream),
          "cudaMemcpyAsync topology upload");
  }
}

__device__ std::size_t NodeIndex(const Params &params, std::size_t batch,
                                 std::size_t node, std::size_t state) {
  return (node * params.batch + batch) * params.states + state;
}

__device__ std::size_t PathIndex(const Params &params, std::size_t batch,
                                 std::size_t edge, std::size_t parent_state,
                                 std::size_t child_state) {
  const std::size_t path_batch = params.paths_batched ? batch : 0;
  const std::size_t path_batches = params.paths_batched ? params.batch : 1;
  return ((edge * path_batches + path_batch) * params.states + parent_state) *
             params.states +
         child_state;
}

__device__ std::size_t BranchIndex(const Params &params, std::size_t batch,
                                   std::size_t branch, std::size_t state) {
  return (branch * params.batch + batch) * params.states + state;
}

__device__ std::size_t NodeScaleIndex(const Params &params, std::size_t batch,
                                      std::size_t node) {
  return node * params.batch + batch;
}

__device__ std::size_t PathScaleIndex(const Params &params, std::size_t batch,
                                      std::size_t edge) {
  const std::size_t path_batch = params.paths_batched ? batch : 0;
  const std::size_t path_batches = params.paths_batched ? params.batch : 1;
  return edge * path_batches + path_batch;
}

__device__ std::size_t BranchScaleIndex(const Params &params, std::size_t batch,
                                        std::size_t branch) {
  return branch * params.batch + batch;
}

__device__ std::size_t AssignmentIndex(const Params &params, std::size_t batch,
                                       std::size_t node) {
  return batch * params.nodes + node;
}

__device__ std::size_t RakeChoiceIndex(const Params &params, std::size_t batch,
                                       std::size_t branch,
                                       std::size_t parent_state) {
  return (branch * params.batch + batch) * params.states + parent_state;
}

__device__ std::size_t CompressionChoiceIndex(const Params &params,
                                              std::size_t batch,
                                              std::size_t tape,
                                              std::size_t parent_state,
                                              std::size_t child_state) {
  return ((tape * params.batch + batch) * params.states + parent_state) *
             params.states +
         child_state;
}

// Transpose the public [batch, node, state] input into the internal
// [node, batch, state] layout without constraining the state count.
constexpr std::size_t kTransposeTile = 32;
constexpr std::size_t kTransposeRows = 8;

__global__ void InitializeNodes(const float *input, float *nodes,
                                Params params) {
  __shared__ float tile[kTransposeTile][kTransposeTile + 1];
  const std::size_t node = blockIdx.x * kTransposeTile + threadIdx.x;
  const std::size_t batch_base = blockIdx.y * kTransposeTile;
  for (std::size_t state = 0; state < params.states; ++state) {
    for (std::size_t row = threadIdx.y; row < kTransposeTile;
         row += kTransposeRows) {
      const std::size_t batch = batch_base + row;
      if (node < params.nodes && batch < params.batch) {
        const std::size_t input_index =
            (batch * params.nodes + node) * params.states + state;
        tile[row][threadIdx.x] = input[input_index];
      }
    }
    __syncthreads();
    const std::size_t batch = batch_base + threadIdx.x;
    for (std::size_t row = threadIdx.y; row < kTransposeTile;
         row += kTransposeRows) {
      const std::size_t output_node = blockIdx.x * kTransposeTile + row;
      if (output_node < params.nodes && batch < params.batch) {
        nodes[NodeIndex(params, batch, output_node, state)] =
            tile[threadIdx.x][row];
      }
    }
    __syncthreads();
  }
}

__global__ void
InitializeCategoricalBase(const float *root_potential,
                          const btrc::Index *observation_index_by_node,
                          float *nodes, Params params) {
  const std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  const std::size_t count =
      static_cast<std::size_t>(params.nodes) * params.batch;
  if (index >= count)
    return;
  const std::size_t batch = index % params.batch;
  const std::size_t node = index / params.batch;
  if (observation_index_by_node[node] !=
      std::numeric_limits<btrc::Index>::max())
    return;
  for (std::size_t state = 0; state < params.states; ++state) {
    nodes[NodeIndex(params, batch, node, state)] =
        node == params.root ? root_potential[state] : 1.0f;
  }
}

__global__ void ApplyCategoricalObservations(
    const std::uint8_t *observations, const btrc::Index *observation_nodes,
    const float *root_potential, const float *emission_potentials, float *nodes,
    Params params, std::uint32_t observation_count, std::uint32_t categories) {
  __shared__ std::uint8_t tile[kTransposeTile][kTransposeTile + 1];
  const std::size_t observation = blockIdx.x * kTransposeTile + threadIdx.x;
  const std::size_t batch_base = blockIdx.y * kTransposeTile;
  for (std::size_t row = threadIdx.y; row < kTransposeTile;
       row += kTransposeRows) {
    const std::size_t batch = batch_base + row;
    if (observation < observation_count && batch < params.batch) {
      tile[row][threadIdx.x] =
          observations[batch * observation_count + observation];
    }
  }
  __syncthreads();
  const std::size_t batch = batch_base + threadIdx.x;
  for (std::size_t row = threadIdx.y; row < kTransposeTile;
       row += kTransposeRows) {
    const std::size_t output_observation = blockIdx.x * kTransposeTile + row;
    if (output_observation >= observation_count || batch >= params.batch)
      continue;
    const std::uint8_t category = tile[threadIdx.x][row];
    const btrc::Index node = observation_nodes[output_observation];
    for (std::size_t state = 0; state < params.states; ++state) {
      const float emission =
          category < categories
              ? emission_potentials[static_cast<std::size_t>(category) *
                                        params.states +
                                    state]
              : 0.0f;
      nodes[NodeIndex(params, batch, node, state)] =
          (node == params.root ? root_potential[state] : 1.0f) * emission;
    }
  }
}

__global__ void InitializePaths(const float *input, float *paths,
                                Params params) {
  const std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  const std::size_t path_batches = params.paths_batched ? params.batch : 1;
  const std::size_t count =
      static_cast<std::size_t>(params.edges) * path_batches;
  if (index >= count)
    return;
  const std::size_t batch = index % path_batches;
  const std::size_t edge = index / path_batches;
  const std::size_t matrix =
      static_cast<std::size_t>(params.states) * params.states;
  const std::size_t output = PathIndex(params, batch, edge, 0, 0);
  const std::size_t source = edge * matrix;
  for (std::size_t entry = 0; entry < matrix; ++entry)
    paths[output + entry] = input[source + entry];
}

__global__ void TakeLogs(float *values, std::size_t count) {
  const std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < count)
    values[index] = values[index] > 0.0f ? logf(values[index]) : -INFINITY;
}

__global__ void MaximumRakeSerial(const btrc::Rake *operations,
                                  const float *nodes, const float *paths,
                                  float *branches, std::uint32_t *choices,
                                  Params params) {
  const std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  const std::size_t count =
      static_cast<std::size_t>(params.batch) * params.operation_count;
  if (index >= count)
    return;
  const std::size_t batch = index % params.batch;
  const std::size_t operation_in_batch = index / params.batch;
  const btrc::Rake operation =
      operations[params.operation_offset + operation_in_batch];
  const std::size_t node_base = NodeIndex(params, batch, operation.leaf, 0);
  const std::size_t path_base = PathIndex(params, batch, operation.edge, 0, 0);
  for (std::size_t parent_state = 0; parent_state < params.states;
       ++parent_state) {
    float best =
        paths[path_base + parent_state * params.states] + nodes[node_base];
    std::uint32_t choice = 0;
    for (std::uint32_t child_state = 1; child_state < params.states;
         ++child_state) {
      const float candidate =
          paths[path_base + parent_state * params.states + child_state] +
          nodes[node_base + child_state];
      if (candidate > best) {
        best = candidate;
        choice = child_state;
      }
    }
    branches[BranchIndex(params, batch, operation.branch, parent_state)] = best;
    choices[RakeChoiceIndex(params, batch, operation.branch, parent_state)] =
        choice;
  }
}

__global__ void MaximumRake(const btrc::Rake *operations, const float *nodes,
                            const float *paths, float *branches,
                            std::uint32_t *choices, Params params) {
  const std::size_t parent_state = threadIdx.x;
  const std::size_t batch = blockIdx.x % params.batch;
  const std::size_t operation_in_batch = blockIdx.x / params.batch;
  if (parent_state >= params.states || batch >= params.batch)
    return;
  const btrc::Rake operation =
      operations[params.operation_offset + operation_in_batch];
  const std::size_t node_base = NodeIndex(params, batch, operation.leaf, 0);
  const std::size_t path_base = PathIndex(params, batch, operation.edge, 0, 0);
  float best =
      paths[path_base + parent_state * params.states] + nodes[node_base];
  std::uint32_t choice = 0;
  for (std::uint32_t child_state = 1; child_state < params.states;
       ++child_state) {
    const float candidate =
        paths[path_base + parent_state * params.states + child_state] +
        nodes[node_base + child_state];
    if (candidate > best) {
      best = candidate;
      choice = child_state;
    }
  }
  branches[BranchIndex(params, batch, operation.branch, parent_state)] = best;
  choices[RakeChoiceIndex(params, batch, operation.branch, parent_state)] =
      choice;
}

__global__ void
MaximumCombineBranchesSerial(const btrc::BranchCombination *operations,
                             float *branches, Params params) {
  const std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  const std::size_t count =
      static_cast<std::size_t>(params.batch) * params.operation_count;
  if (index >= count)
    return;
  const std::size_t batch = index % params.batch;
  const std::size_t operation_in_batch = index / params.batch;
  const btrc::BranchCombination operation =
      operations[params.operation_offset + operation_in_batch];
  for (std::size_t state = 0; state < params.states; ++state) {
    branches[BranchIndex(params, batch, operation.destination, state)] +=
        branches[BranchIndex(params, batch, operation.source, state)];
  }
}

__global__ void
MaximumCombineBranches(const btrc::BranchCombination *operations,
                       float *branches, Params params) {
  const std::size_t state = threadIdx.x;
  const std::size_t batch = blockIdx.x % params.batch;
  const std::size_t operation_in_batch = blockIdx.x / params.batch;
  if (state >= params.states || batch >= params.batch)
    return;
  const btrc::BranchCombination operation =
      operations[params.operation_offset + operation_in_batch];
  branches[BranchIndex(params, batch, operation.destination, state)] +=
      branches[BranchIndex(params, batch, operation.source, state)];
}

__global__ void
MaximumAbsorbBranchesSerial(const btrc::BranchAbsorption *operations,
                            float *nodes, const float *branches,
                            Params params) {
  const std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  const std::size_t count =
      static_cast<std::size_t>(params.batch) * params.operation_count;
  if (index >= count)
    return;
  const std::size_t batch = index % params.batch;
  const std::size_t operation_in_batch = index / params.batch;
  const btrc::BranchAbsorption operation =
      operations[params.operation_offset + operation_in_batch];
  for (std::size_t state = 0; state < params.states; ++state) {
    nodes[NodeIndex(params, batch, operation.parent, state)] +=
        branches[BranchIndex(params, batch, operation.branch, state)];
  }
}

__global__ void MaximumAbsorbBranches(const btrc::BranchAbsorption *operations,
                                      float *nodes, const float *branches,
                                      Params params) {
  const std::size_t state = threadIdx.x;
  const std::size_t batch = blockIdx.x % params.batch;
  const std::size_t operation_in_batch = blockIdx.x / params.batch;
  if (state >= params.states || batch >= params.batch)
    return;
  const btrc::BranchAbsorption operation =
      operations[params.operation_offset + operation_in_batch];
  nodes[NodeIndex(params, batch, operation.parent, state)] +=
      branches[BranchIndex(params, batch, operation.branch, state)];
}

__global__ void MaximumCompressSerial4(const btrc::Compression *operations,
                                       const float *nodes, float *paths,
                                       std::uint32_t *choices, Params params) {
  const std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  const std::size_t count =
      static_cast<std::size_t>(params.batch) * params.operation_count;
  if (index >= count)
    return;
  const std::size_t batch = index % params.batch;
  const std::size_t operation_in_batch = index / params.batch;
  const btrc::Compression operation =
      operations[params.operation_offset + operation_in_batch];
  constexpr std::size_t kStates = 4;
  constexpr std::size_t kMatrix = kStates * kStates;
  float left[kMatrix];
  float right[kMatrix];
  float middle[kStates];
  const std::size_t left_base =
      PathIndex(params, batch, operation.left_edge, 0, 0);
  const std::size_t right_base =
      PathIndex(params, batch, operation.right_edge, 0, 0);
  const std::size_t middle_base = NodeIndex(params, batch, operation.middle, 0);
  for (std::size_t entry = 0; entry < kMatrix; ++entry) {
    left[entry] = paths[left_base + entry];
    right[entry] = paths[right_base + entry];
  }
  for (std::size_t state = 0; state < kStates; ++state)
    middle[state] = nodes[middle_base + state];
  for (std::size_t parent_state = 0; parent_state < kStates; ++parent_state) {
    for (std::size_t child_state = 0; child_state < kStates; ++child_state) {
      float best =
          left[parent_state * kStates] + middle[0] + right[child_state];
      std::uint32_t choice = 0;
      for (std::uint32_t middle_state = 1; middle_state < kStates;
           ++middle_state) {
        const float candidate = left[parent_state * kStates + middle_state] +
                                middle[middle_state] +
                                right[middle_state * kStates + child_state];
        if (candidate > best) {
          best = candidate;
          choice = middle_state;
        }
      }
      paths[left_base + parent_state * kStates + child_state] = best;
      choices[CompressionChoiceIndex(params, batch, operation.tape,
                                     parent_state, child_state)] = choice;
    }
  }
}

__global__ void MaximumCompress(const btrc::Compression *operations,
                                const float *nodes, float *paths,
                                std::uint32_t *choices, Params params) {
  extern __shared__ float storage[];
  const std::size_t matrix =
      static_cast<std::size_t>(params.states) * params.states;
  float *left = storage;
  float *right = left + matrix;
  float *middle = right + matrix;
  const std::size_t batch = blockIdx.x % params.batch;
  const std::size_t operation_in_batch = blockIdx.x / params.batch;
  const std::size_t entry = threadIdx.x;
  if (batch >= params.batch)
    return;
  const btrc::Compression operation =
      operations[params.operation_offset + operation_in_batch];
  if (entry < matrix) {
    const std::size_t parent_state = entry / params.states;
    const std::size_t child_state = entry % params.states;
    left[entry] = paths[PathIndex(params, batch, operation.left_edge,
                                  parent_state, child_state)];
    right[entry] = paths[PathIndex(params, batch, operation.right_edge,
                                   parent_state, child_state)];
  }
  if (entry < params.states)
    middle[entry] = nodes[NodeIndex(params, batch, operation.middle, entry)];
  __syncthreads();
  if (entry >= matrix)
    return;

  const std::size_t parent_state = entry / params.states;
  const std::size_t child_state = entry % params.states;
  float best =
      left[parent_state * params.states] + middle[0] + right[child_state];
  std::uint32_t choice = 0;
  for (std::uint32_t middle_state = 1; middle_state < params.states;
       ++middle_state) {
    const float candidate = left[parent_state * params.states + middle_state] +
                            middle[middle_state] +
                            right[middle_state * params.states + child_state];
    if (candidate > best) {
      best = candidate;
      choice = middle_state;
    }
  }
  paths[PathIndex(params, batch, operation.left_edge, parent_state,
                  child_state)] = best;
  choices[CompressionChoiceIndex(params, batch, operation.tape, parent_state,
                                 child_state)] = choice;
}

__global__ void FinishMaximum(const float *nodes, float *log_weights,
                              std::uint32_t *assignments, Params params) {
  const std::size_t batch = blockIdx.x * blockDim.x + threadIdx.x;
  if (batch >= params.batch)
    return;
  const std::size_t root_base = NodeIndex(params, batch, params.root, 0);
  float best = nodes[root_base];
  std::uint32_t choice = 0;
  for (std::uint32_t state = 1; state < params.states; ++state) {
    if (nodes[root_base + state] > best) {
      best = nodes[root_base + state];
      choice = state;
    }
  }
  log_weights[batch] = best;
  assignments[AssignmentIndex(params, batch, params.root)] = choice;
}

__global__ void ExpandMaximumRakes(const btrc::Rake *operations,
                                   const std::uint32_t *choices,
                                   std::uint32_t *assignments, Params params) {
  const std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  const std::size_t count =
      static_cast<std::size_t>(params.batch) * params.operation_count;
  if (index >= count)
    return;
  const std::size_t batch = index % params.batch;
  const std::size_t operation_in_batch = index / params.batch;
  const btrc::Rake operation =
      operations[params.operation_offset + operation_in_batch];
  const std::uint32_t parent_state =
      assignments[AssignmentIndex(params, batch, operation.parent)];
  assignments[AssignmentIndex(params, batch, operation.leaf)] =
      choices[RakeChoiceIndex(params, batch, operation.branch, parent_state)];
}

__global__ void ExpandMaximumCompressions(const btrc::Compression *operations,
                                          const std::uint32_t *choices,
                                          std::uint32_t *assignments,
                                          Params params) {
  const std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  const std::size_t count =
      static_cast<std::size_t>(params.batch) * params.operation_count;
  if (index >= count)
    return;
  const std::size_t batch = index % params.batch;
  const std::size_t operation_in_batch = index / params.batch;
  const btrc::Compression operation =
      operations[params.operation_offset + operation_in_batch];
  const std::uint32_t parent_state =
      assignments[AssignmentIndex(params, batch, operation.parent)];
  const std::uint32_t child_state =
      assignments[AssignmentIndex(params, batch, operation.child)];
  assignments[AssignmentIndex(params, batch, operation.middle)] =
      choices[CompressionChoiceIndex(params, batch, operation.tape,
                                     parent_state, child_state)];
}

__global__ void Rake(const btrc::Rake *operations, const float *nodes,
                     const float *paths, float *branches,
                     const float *node_scales, const float *path_scales,
                     float *branch_scales, Params params) {
  __shared__ float normalizer;
  const std::size_t state = threadIdx.x;
  const std::size_t batch = blockIdx.x % params.batch;
  const std::size_t operation_in_batch = blockIdx.x / params.batch;
  const btrc::Rake operation =
      operations[params.operation_offset + operation_in_batch];
  if (state >= params.states || batch >= params.batch)
    return;
  const float value =
      detail::RakeValue(paths + PathIndex(params, batch, operation.edge, 0, 0),
                        nodes + NodeIndex(params, batch, operation.leaf, 0),
                        params.states, state);
  const std::size_t branch_base =
      BranchIndex(params, batch, operation.branch, 0);
  branches[branch_base + state] = value;
  if (!params.scaled)
    return;
  __syncthreads();
  if (state == 0) {
    const float maximum =
        detail::Maximum(branches + branch_base, params.states);
    normalizer = maximum > 0.0f ? maximum : 1.0f;
    const float input_scale =
        node_scales[NodeScaleIndex(params, batch, operation.leaf)] +
        path_scales[PathScaleIndex(params, batch, operation.edge)];
    branch_scales[BranchScaleIndex(params, batch, operation.branch)] =
        detail::UpdatedLogScale(input_scale, maximum);
  }
  __syncthreads();
  branches[branch_base + state] = value / normalizer;
}

__global__ void RakeSerial(const btrc::Rake *operations, const float *nodes,
                           const float *paths, float *branches,
                           const float *node_scales, const float *path_scales,
                           float *branch_scales, Params params) {
  const std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  const std::size_t count =
      static_cast<std::size_t>(params.batch) * params.operation_count;
  if (index >= count)
    return;
  const std::size_t batch = index % params.batch;
  const std::size_t operation_in_batch = index / params.batch;
  const btrc::Rake operation =
      operations[params.operation_offset + operation_in_batch];
  const std::size_t node_base = NodeIndex(params, batch, operation.leaf, 0);
  const std::size_t path_base = PathIndex(params, batch, operation.edge, 0, 0);
  const std::size_t branch_base =
      BranchIndex(params, batch, operation.branch, 0);
  float maximum = 0.0f;
  for (std::size_t parent_state = 0; parent_state < params.states;
       ++parent_state) {
    const float value = detail::RakeValue(paths + path_base, nodes + node_base,
                                          params.states, parent_state);
    branches[branch_base + parent_state] = value;
    maximum = fmaxf(maximum, value);
  }
  if (!params.scaled)
    return;
  const float normalizer = maximum > 0.0f ? maximum : 1.0f;
  for (std::size_t state = 0; state < params.states; ++state)
    branches[branch_base + state] /= normalizer;
  const float input_scale =
      node_scales[NodeScaleIndex(params, batch, operation.leaf)] +
      path_scales[PathScaleIndex(params, batch, operation.edge)];
  branch_scales[BranchScaleIndex(params, batch, operation.branch)] =
      detail::UpdatedLogScale(input_scale, maximum);
}

__global__ void CombineBranches(const btrc::BranchCombination *operations,
                                float *branches, float *branch_scales,
                                Params params) {
  __shared__ float normalizer;
  const std::size_t state = threadIdx.x;
  const std::size_t batch = blockIdx.x % params.batch;
  const std::size_t operation_in_batch = blockIdx.x / params.batch;
  const btrc::BranchCombination operation =
      operations[params.operation_offset + operation_in_batch];
  if (state >= params.states || batch >= params.batch)
    return;
  const std::size_t destination_base =
      BranchIndex(params, batch, operation.destination, 0);
  const std::size_t source_base =
      BranchIndex(params, batch, operation.source, 0);
  const float value = detail::Product(branches[destination_base + state],
                                      branches[source_base + state]);
  branches[destination_base + state] = value;
  if (!params.scaled)
    return;
  __syncthreads();
  if (state == 0) {
    const float maximum =
        detail::Maximum(branches + destination_base, params.states);
    normalizer = maximum > 0.0f ? maximum : 1.0f;
    const std::size_t destination_scale =
        BranchScaleIndex(params, batch, operation.destination);
    const std::size_t source_scale =
        BranchScaleIndex(params, batch, operation.source);
    const float input_scale =
        branch_scales[destination_scale] + branch_scales[source_scale];
    branch_scales[destination_scale] =
        detail::UpdatedLogScale(input_scale, maximum);
  }
  __syncthreads();
  branches[destination_base + state] = value / normalizer;
}

__global__ void CombineBranchesSerial(const btrc::BranchCombination *operations,
                                      float *branches, float *branch_scales,
                                      Params params) {
  const std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  const std::size_t count =
      static_cast<std::size_t>(params.batch) * params.operation_count;
  if (index >= count)
    return;
  const std::size_t batch = index % params.batch;
  const std::size_t operation_in_batch = index / params.batch;
  const btrc::BranchCombination operation =
      operations[params.operation_offset + operation_in_batch];
  const std::size_t destination_base =
      BranchIndex(params, batch, operation.destination, 0);
  const std::size_t source_base =
      BranchIndex(params, batch, operation.source, 0);
  float maximum = 0.0f;
  for (std::size_t state = 0; state < params.states; ++state) {
    const float value = detail::Product(branches[destination_base + state],
                                        branches[source_base + state]);
    branches[destination_base + state] = value;
    maximum = fmaxf(maximum, value);
  }
  if (!params.scaled)
    return;
  const float normalizer = maximum > 0.0f ? maximum : 1.0f;
  for (std::size_t state = 0; state < params.states; ++state)
    branches[destination_base + state] /= normalizer;
  const std::size_t destination_scale =
      BranchScaleIndex(params, batch, operation.destination);
  const std::size_t source_scale =
      BranchScaleIndex(params, batch, operation.source);
  const float input_scale =
      branch_scales[destination_scale] + branch_scales[source_scale];
  branch_scales[destination_scale] =
      detail::UpdatedLogScale(input_scale, maximum);
}

__global__ void AbsorbBranches(const btrc::BranchAbsorption *operations,
                               float *nodes, const float *branches,
                               float *node_scales, const float *branch_scales,
                               Params params) {
  __shared__ float normalizer;
  const std::size_t state = threadIdx.x;
  const std::size_t batch = blockIdx.x % params.batch;
  const std::size_t operation_in_batch = blockIdx.x / params.batch;
  const btrc::BranchAbsorption operation =
      operations[params.operation_offset + operation_in_batch];
  if (state >= params.states || batch >= params.batch)
    return;
  const std::size_t node_base = NodeIndex(params, batch, operation.parent, 0);
  const std::size_t branch_base =
      BranchIndex(params, batch, operation.branch, 0);
  const float value =
      detail::Product(nodes[node_base + state], branches[branch_base + state]);
  nodes[node_base + state] = value;
  if (!params.scaled)
    return;
  __syncthreads();
  if (state == 0) {
    const float maximum = detail::Maximum(nodes + node_base, params.states);
    normalizer = maximum > 0.0f ? maximum : 1.0f;
    const std::size_t node_scale =
        NodeScaleIndex(params, batch, operation.parent);
    const float input_scale =
        node_scales[node_scale] +
        branch_scales[BranchScaleIndex(params, batch, operation.branch)];
    node_scales[node_scale] = detail::UpdatedLogScale(input_scale, maximum);
  }
  __syncthreads();
  nodes[node_base + state] = value / normalizer;
}

__global__ void AbsorbBranchesSerial(const btrc::BranchAbsorption *operations,
                                     float *nodes, const float *branches,
                                     float *node_scales,
                                     const float *branch_scales,
                                     Params params) {
  const std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  const std::size_t count =
      static_cast<std::size_t>(params.batch) * params.operation_count;
  if (index >= count)
    return;
  const std::size_t batch = index % params.batch;
  const std::size_t operation_in_batch = index / params.batch;
  const btrc::BranchAbsorption operation =
      operations[params.operation_offset + operation_in_batch];
  const std::size_t node_base = NodeIndex(params, batch, operation.parent, 0);
  const std::size_t branch_base =
      BranchIndex(params, batch, operation.branch, 0);
  float maximum = 0.0f;
  for (std::size_t state = 0; state < params.states; ++state) {
    const float value = detail::Product(nodes[node_base + state],
                                        branches[branch_base + state]);
    nodes[node_base + state] = value;
    maximum = fmaxf(maximum, value);
  }
  if (!params.scaled)
    return;
  const float normalizer = maximum > 0.0f ? maximum : 1.0f;
  for (std::size_t state = 0; state < params.states; ++state)
    nodes[node_base + state] /= normalizer;
  const std::size_t node_scale =
      NodeScaleIndex(params, batch, operation.parent);
  const float input_scale =
      node_scales[node_scale] +
      branch_scales[BranchScaleIndex(params, batch, operation.branch)];
  node_scales[node_scale] = detail::UpdatedLogScale(input_scale, maximum);
}

__global__ void Compress(const btrc::Compression *operations,
                         const float *nodes, float *paths,
                         const float *node_scales, float *path_scales,
                         Params params) {
  extern __shared__ float storage[];
  __shared__ float normalizer;
  const std::size_t matrix =
      static_cast<std::size_t>(params.states) * params.states;
  float *left = storage;
  float *right = left + matrix;
  float *middle = right + matrix;
  const std::size_t batch = blockIdx.x % params.batch;
  const std::size_t operation_in_batch = blockIdx.x / params.batch;
  const btrc::Compression operation =
      operations[params.operation_offset + operation_in_batch];
  const std::size_t entry = threadIdx.x;
  if (batch >= params.batch)
    return;
  if (entry < matrix) {
    const std::size_t parent_state = entry / params.states;
    const std::size_t child_state = entry % params.states;
    left[entry] = paths[PathIndex(params, batch, operation.left_edge,
                                  parent_state, child_state)];
    right[entry] = paths[PathIndex(params, batch, operation.right_edge,
                                   parent_state, child_state)];
  }
  if (entry < params.states) {
    middle[entry] = nodes[NodeIndex(params, batch, operation.middle, entry)];
  }
  __syncthreads();
  if (entry >= matrix)
    return;
  const std::size_t parent_state = entry / params.states;
  const std::size_t child_state = entry % params.states;
  const float value = detail::CompressionValue(
      left, middle, right, params.states, parent_state, child_state);
  paths[PathIndex(params, batch, operation.left_edge, parent_state,
                  child_state)] = value;
  if (!params.scaled)
    return;
  __syncthreads();
  if (entry == 0) {
    const std::size_t output_base =
        PathIndex(params, batch, operation.left_edge, 0, 0);
    const float maximum =
        detail::Maximum(paths + output_base, static_cast<unsigned>(matrix));
    normalizer = maximum > 0.0f ? maximum : 1.0f;
    const std::size_t left_scale =
        PathScaleIndex(params, batch, operation.left_edge);
    const float input_scale =
        path_scales[left_scale] +
        node_scales[NodeScaleIndex(params, batch, operation.middle)] +
        path_scales[PathScaleIndex(params, batch, operation.right_edge)];
    path_scales[left_scale] = detail::UpdatedLogScale(input_scale, maximum);
  }
  __syncthreads();
  paths[PathIndex(params, batch, operation.left_edge, parent_state,
                  child_state)] = value / normalizer;
}

__global__ void CompressSerial4(const btrc::Compression *operations,
                                const float *nodes, float *paths,
                                const float *node_scales, float *path_scales,
                                Params params) {
  const std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  const std::size_t count =
      static_cast<std::size_t>(params.batch) * params.operation_count;
  if (index >= count)
    return;
  const std::size_t batch = index % params.batch;
  const std::size_t operation_in_batch = index / params.batch;
  const btrc::Compression operation =
      operations[params.operation_offset + operation_in_batch];
  constexpr std::size_t kStates = 4;
  constexpr std::size_t kMatrix = kStates * kStates;
  float left[kMatrix];
  float right[kMatrix];
  float middle[kStates];
  for (std::size_t entry = 0; entry < kMatrix; ++entry) {
    const std::size_t parent_state = entry / kStates;
    const std::size_t child_state = entry % kStates;
    left[entry] = paths[PathIndex(params, batch, operation.left_edge,
                                  parent_state, child_state)];
    right[entry] = paths[PathIndex(params, batch, operation.right_edge,
                                   parent_state, child_state)];
  }
  for (std::size_t state = 0; state < kStates; ++state)
    middle[state] = nodes[NodeIndex(params, batch, operation.middle, state)];

  float maximum = 0.0f;
  for (std::size_t parent_state = 0; parent_state < kStates; ++parent_state) {
    for (std::size_t child_state = 0; child_state < kStates; ++child_state) {
      const float value = detail::CompressionValue(left, middle, right, kStates,
                                                   parent_state, child_state);
      paths[PathIndex(params, batch, operation.left_edge, parent_state,
                      child_state)] = value;
      maximum = fmaxf(maximum, value);
    }
  }
  if (!params.scaled)
    return;
  const float normalizer = maximum > 0.0f ? maximum : 1.0f;
  for (std::size_t entry = 0; entry < kMatrix; ++entry) {
    const std::size_t parent_state = entry / kStates;
    const std::size_t child_state = entry % kStates;
    paths[PathIndex(params, batch, operation.left_edge, parent_state,
                    child_state)] /= normalizer;
  }
  const std::size_t left_scale =
      PathScaleIndex(params, batch, operation.left_edge);
  const float input_scale =
      path_scales[left_scale] +
      node_scales[NodeScaleIndex(params, batch, operation.middle)] +
      path_scales[PathScaleIndex(params, batch, operation.right_edge)];
  path_scales[left_scale] = detail::UpdatedLogScale(input_scale, maximum);
}

__global__ void FinishRoot(const float *nodes, const float *node_scales,
                           float *output, Params params) {
  const std::size_t batch = blockIdx.x;
  if (batch >= params.batch || threadIdx.x != 0)
    return;
  float value = 0.0f;
  for (std::size_t state = 0; state < params.states; ++state)
    value += nodes[NodeIndex(params, batch, params.root, state)];
  output[batch] =
      params.scaled
          ? (value > 0.0f
                 ? node_scales[NodeScaleIndex(params, batch, params.root)] +
                       logf(value)
                 : -INFINITY)
          : value;
}

std::uint32_t Blocks(std::size_t count, std::size_t block_size) {
  return CheckedU32((count + block_size - 1) / block_size, "CUDA grid size");
}

} // namespace

struct Workspace::Impl {
  const btrc::Plan *plan = nullptr;
  std::size_t states = 0;
  std::size_t batch = 0;
  int device = 0;
  Params params{};
  cudaStream_t stream = nullptr;
  cudaEvent_t upload_start = nullptr;
  cudaEvent_t upload_stop = nullptr;
  cudaEvent_t kernel_start = nullptr;
  cudaEvent_t kernel_stop = nullptr;
  cudaEvent_t download_start = nullptr;
  cudaEvent_t download_stop = nullptr;

  btrc::Rake *rakes = nullptr;
  btrc::BranchCombination *combinations = nullptr;
  btrc::BranchAbsorption *absorptions = nullptr;
  btrc::Compression *compressions = nullptr;
  float *host_nodes = nullptr;
  float *host_edges = nullptr;
  float *host_output = nullptr;
  std::uint32_t *host_assignments = nullptr;
  std::uint8_t *host_observations = nullptr;
  float *host_root_potential = nullptr;
  float *host_emission_potentials = nullptr;
  float *input_nodes = nullptr;
  float *input_edges = nullptr;
  std::uint8_t *input_observations = nullptr;
  float *input_root_potential = nullptr;
  float *input_emission_potentials = nullptr;
  btrc::Index *categorical_observation_nodes = nullptr;
  btrc::Index *categorical_observation_index_by_node = nullptr;
  float *nodes = nullptr;
  float *paths = nullptr;
  float *branches = nullptr;
  float *node_scales = nullptr;
  float *path_scales = nullptr;
  float *branch_scales = nullptr;
  float *output = nullptr;
  std::uint32_t *rake_choices = nullptr;
  std::uint32_t *compression_choices = nullptr;
  std::uint32_t *assignments = nullptr;
  std::vector<btrc::Index> observation_nodes;
  std::size_t categories = 0;
  bool categorical = false;
  bool bidirectional = false;

  void Clear() noexcept {
    if (stream != nullptr) {
      static_cast<void>(cudaSetDevice(device));
      static_cast<void>(cudaStreamSynchronize(stream));
    }
    DeviceFree(rakes);
    DeviceFree(combinations);
    DeviceFree(absorptions);
    DeviceFree(compressions);
    DeviceFree(input_nodes);
    DeviceFree(input_edges);
    DeviceFree(input_observations);
    DeviceFree(input_root_potential);
    DeviceFree(input_emission_potentials);
    DeviceFree(categorical_observation_nodes);
    DeviceFree(categorical_observation_index_by_node);
    DeviceFree(nodes);
    DeviceFree(paths);
    DeviceFree(branches);
    DeviceFree(node_scales);
    DeviceFree(path_scales);
    DeviceFree(branch_scales);
    DeviceFree(output);
    DeviceFree(rake_choices);
    DeviceFree(compression_choices);
    DeviceFree(assignments);
    HostFree(host_nodes);
    HostFree(host_edges);
    HostFree(host_output);
    HostFree(host_assignments);
    HostFree(host_observations);
    HostFree(host_root_potential);
    HostFree(host_emission_potentials);
    Destroy(upload_start);
    Destroy(upload_stop);
    Destroy(kernel_start);
    Destroy(kernel_stop);
    Destroy(download_start);
    Destroy(download_stop);
    Destroy(stream);
    plan = nullptr;
    states = 0;
    batch = 0;
    device = 0;
    params = {};
    observation_nodes.clear();
    categories = 0;
    categorical = false;
    bidirectional = false;
  }

  ~Impl() { Clear(); }
};

Workspace::Workspace() : impl_(std::make_unique<Impl>()) {}
Workspace::~Workspace() = default;
Workspace::Workspace(Workspace &&) noexcept = default;
Workspace &Workspace::operator=(Workspace &&) noexcept = default;

bool Available() {
  int count = 0;
  return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
}

std::string DeviceDescription(int device) {
  cudaDeviceProp properties{};
  Check(cudaGetDeviceProperties(&properties, device),
        "cudaGetDeviceProperties");
  std::ostringstream result;
  result << properties.name << " (compute " << properties.major << '.'
         << properties.minor << ')';
  return result.str();
}

namespace {

void ReserveCommon(Workspace::Impl &storage, const btrc::Plan &plan,
                   std::size_t states, std::size_t batch, int device) {
  if (states == 0 || batch == 0)
    throw std::invalid_argument("CUDA state and batch counts must be nonzero");
  storage.Clear();
  Check(cudaSetDevice(device), "cudaSetDevice");
  cudaDeviceProp properties{};
  Check(cudaGetDeviceProperties(&properties, device),
        "cudaGetDeviceProperties");
  const std::size_t matrix = CheckedProduct({states, states}, "state matrix");
  if (matrix > static_cast<std::size_t>(properties.maxThreadsPerBlock)) {
    throw std::invalid_argument(
        "state count exceeds the CUDA compression block limit");
  }

  storage.plan = &plan;
  storage.states = states;
  storage.batch = batch;
  storage.device = device;
  storage.params = {
      CheckedU32(states, "state count"),
      CheckedU32(plan.num_nodes(), "node count"),
      CheckedU32(plan.num_edges(), "edge count"),
      CheckedU32(plan.num_branches(), "branch count"),
      CheckedU32(batch, "batch count"),
      plan.root(),
      0,
      0,
      0,
      plan.num_compressions() == 0 ? 0U : 1U,
  };

  Check(cudaStreamCreateWithFlags(&storage.stream, cudaStreamNonBlocking),
        "cudaStreamCreateWithFlags");
  Check(cudaEventCreate(&storage.upload_start), "cudaEventCreate");
  Check(cudaEventCreate(&storage.upload_stop), "cudaEventCreate");
  Check(cudaEventCreate(&storage.kernel_start), "cudaEventCreate");
  Check(cudaEventCreate(&storage.kernel_stop), "cudaEventCreate");
  Check(cudaEventCreate(&storage.download_start), "cudaEventCreate");
  Check(cudaEventCreate(&storage.download_stop), "cudaEventCreate");

  DeviceAllocate(storage.rakes, plan.rakes().size());
  DeviceAllocate(storage.combinations, plan.branch_combinations().size());
  DeviceAllocate(storage.absorptions, plan.branch_absorptions().size());
  DeviceAllocate(storage.compressions, plan.compressions().size());
  Upload(storage.rakes, plan.rakes(), storage.stream);
  Upload(storage.combinations, plan.branch_combinations(), storage.stream);
  Upload(storage.absorptions, plan.branch_absorptions(), storage.stream);
  Upload(storage.compressions, plan.compressions(), storage.stream);

  const std::size_t node_values =
      CheckedProduct({batch, plan.num_nodes(), states}, "node workspace");
  const std::size_t edge_inputs =
      CheckedProduct({plan.num_edges(), matrix}, "edge inputs");
  const std::size_t path_batches = plan.num_compressions() == 0 ? 1 : batch;
  const std::size_t path_values = CheckedProduct(
      {path_batches, plan.num_edges(), matrix}, "path workspace");
  const std::size_t branch_values =
      CheckedProduct({batch, plan.num_branches(), states}, "branch workspace");
  HostAllocate(storage.host_edges, edge_inputs);
  HostAllocate(storage.host_output, batch);
  DeviceAllocate(storage.input_edges, edge_inputs);
  DeviceAllocate(storage.nodes, node_values);
  DeviceAllocate(storage.paths, path_values);
  DeviceAllocate(storage.branches, branch_values);
  DeviceAllocate(storage.node_scales,
                 CheckedProduct({batch, plan.num_nodes()}, "node scales"));
  DeviceAllocate(
      storage.path_scales,
      CheckedProduct({path_batches, plan.num_edges()}, "path scales"));
  DeviceAllocate(storage.branch_scales,
                 CheckedProduct({batch, plan.num_branches()}, "branch scales"));
  DeviceAllocate(storage.output, batch);
}

void ReserveRecovery(Workspace::Impl &storage) {
  const btrc::Plan &plan = *storage.plan;
  HostAllocate(storage.host_assignments,
               CheckedProduct({storage.batch, plan.num_nodes()},
                              "host MAP assignments"));
  DeviceAllocate(
      storage.rake_choices,
      CheckedProduct({storage.batch, plan.num_branches(), storage.states},
                     "MAP rake choices"));
  DeviceAllocate(storage.compression_choices,
                 CheckedProduct({storage.batch, plan.num_compressions(),
                                 storage.states, storage.states},
                                "MAP compression choices"));
  DeviceAllocate(
      storage.assignments,
      CheckedProduct({storage.batch, plan.num_nodes()}, "MAP assignments"));
  storage.bidirectional = true;
}

} // namespace

void Workspace::Reserve(const btrc::Plan &plan, std::size_t states,
                        std::size_t batch, int device) {
  Impl &storage = *impl_;
  ReserveCommon(storage, plan, states, batch, device);
  const std::size_t node_values =
      CheckedProduct({batch, plan.num_nodes(), states}, "node inputs");
  HostAllocate(storage.host_nodes, node_values);
  DeviceAllocate(storage.input_nodes, node_values);
  Check(cudaStreamSynchronize(storage.stream), "topology upload");
}

void Workspace::ReserveCategorical(
    const btrc::Plan &plan, std::size_t states, std::size_t batch,
    std::size_t categories, std::span<const btrc::Index> observation_nodes,
    int device) {
  if (categories == 0 || categories > 256)
    throw std::invalid_argument(
        "CUDA categorical model must have between 1 and 256 categories");
  btrc::Index previous = 0;
  bool first = true;
  for (const btrc::Index node : observation_nodes) {
    if (node >= plan.num_nodes() || (!first && node <= previous)) {
      throw std::invalid_argument(
          "CUDA categorical observation nodes must be valid and strictly "
          "increasing");
    }
    previous = node;
    first = false;
  }
  Impl &storage = *impl_;
  ReserveCommon(storage, plan, states, batch, device);
  storage.categorical = true;
  storage.categories = categories;
  storage.observation_nodes.assign(observation_nodes.begin(),
                                   observation_nodes.end());
  const std::size_t observation_values = CheckedProduct(
      {batch, observation_nodes.size()}, "categorical observations");
  const std::size_t emission_values =
      CheckedProduct({categories, states}, "categorical emissions");
  HostAllocate(storage.host_observations, observation_values);
  HostAllocate(storage.host_root_potential, states);
  HostAllocate(storage.host_emission_potentials, emission_values);
  DeviceAllocate(storage.input_observations, observation_values);
  DeviceAllocate(storage.input_root_potential, states);
  DeviceAllocate(storage.input_emission_potentials, emission_values);
  DeviceAllocate(storage.categorical_observation_nodes,
                 observation_nodes.size());
  DeviceAllocate(storage.categorical_observation_index_by_node,
                 plan.num_nodes());
  Upload(storage.categorical_observation_nodes,
         std::span<const btrc::Index>(storage.observation_nodes),
         storage.stream);
  std::vector<btrc::Index> observation_index_by_node(
      plan.num_nodes(), std::numeric_limits<btrc::Index>::max());
  for (std::size_t index = 0; index < observation_nodes.size(); ++index)
    observation_index_by_node[observation_nodes[index]] =
        static_cast<btrc::Index>(index);
  Upload(storage.categorical_observation_index_by_node,
         std::span<const btrc::Index>(observation_index_by_node),
         storage.stream);
  Check(cudaStreamSynchronize(storage.stream), "categorical topology upload");
}

void Workspace::ReserveBidirectional(const btrc::Plan &plan, std::size_t states,
                                     std::size_t batch, int device) {
  Reserve(plan, states, batch, device);
  ReserveRecovery(*impl_);
}

void Workspace::ReserveCategoricalBidirectional(
    const btrc::Plan &plan, std::size_t states, std::size_t batch,
    std::size_t categories, std::span<const btrc::Index> observation_nodes,
    int device) {
  ReserveCategorical(plan, states, batch, categories, observation_nodes,
                     device);
  ReserveRecovery(*impl_);
}

tree_hmm::MutableBatchedModelView Workspace::Inputs() {
  return Inputs(impl_->batch);
}

tree_hmm::MutableBatchedModelView Workspace::Inputs(std::size_t batch) {
  Impl &storage = *impl_;
  if (storage.plan == nullptr)
    throw std::logic_error("CUDA Workspace::Reserve must precede Inputs");
  if (storage.categorical)
    throw std::logic_error(
        "CUDA dense Inputs cannot be used after ReserveCategorical");
  if (batch == 0 || batch > storage.batch)
    throw std::invalid_argument(
        "CUDA input batch exceeds the reserved capacity");
  const std::size_t node_values = CheckedProduct(
      {batch, storage.plan->num_nodes(), storage.states}, "CUDA input nodes");
  const std::size_t edge_values = CheckedProduct(
      {storage.plan->num_edges(), storage.states, storage.states},
      "CUDA input edges");
  return {*storage.plan,
          storage.states,
          batch,
          {storage.host_nodes, node_values},
          {storage.host_edges, edge_values}};
}

tree_hmm::MutableBatchedCategoricalModelView Workspace::CategoricalInputs() {
  return CategoricalInputs(impl_->batch);
}

tree_hmm::MutableBatchedCategoricalModelView
Workspace::CategoricalInputs(std::size_t batch) {
  Impl &storage = *impl_;
  if (storage.plan == nullptr || !storage.categorical) {
    throw std::logic_error(
        "CUDA Workspace::ReserveCategorical must precede CategoricalInputs");
  }
  if (batch == 0 || batch > storage.batch)
    throw std::invalid_argument(
        "CUDA categorical input batch exceeds the reserved capacity");
  const std::size_t observation_values = CheckedProduct(
      {batch, storage.observation_nodes.size()}, "categorical observations");
  const std::size_t emission_values = CheckedProduct(
      {storage.categories, storage.states}, "categorical emissions");
  const std::size_t edge_values = CheckedProduct(
      {storage.plan->num_edges(), storage.states, storage.states},
      "CUDA categorical input edges");
  return {*storage.plan,
          storage.states,
          batch,
          storage.categories,
          storage.observation_nodes,
          {storage.host_observations, observation_values},
          {storage.host_root_potential, storage.states},
          {storage.host_emission_potentials, emission_values},
          {storage.host_edges, edge_values}};
}

namespace {

template <class StageInputs, class UploadInputs, class InitializeNodeData>
tree_hmm::PartitionView
RunPrepared(const btrc::Plan &plan, std::size_t states, std::size_t batch,
            std::span<const float> edge_potentials, Workspace::Impl &storage,
            bool scaled, StageInputs stage_inputs, UploadInputs upload_inputs,
            InitializeNodeData initialize_node_data) {
  const auto wall_start = Clock::now();
  if (storage.plan != &plan || storage.states != states || batch == 0 ||
      batch > storage.batch) {
    throw std::invalid_argument(
        "prepared CUDA inference requires Workspace::Reserve for this plan, "
        "state count, and batch capacity");
  }
  Check(cudaSetDevice(storage.device), "cudaSetDevice");
  const std::size_t matrix = CheckedProduct({states, states}, "state matrix");
  const std::size_t edge_values =
      CheckedProduct({plan.num_edges(), matrix}, "edge inputs");
  if (edge_potentials.size() != edge_values)
    throw std::invalid_argument("CUDA edge input shape is wrong");
  stage_inputs();
  if (edge_potentials.data() != storage.host_edges) {
    std::memcpy(storage.host_edges, edge_potentials.data(),
                edge_potentials.size_bytes());
  }

  Check(cudaEventRecord(storage.upload_start, storage.stream),
        "cudaEventRecord upload start");
  upload_inputs();
  Check(cudaMemcpyAsync(storage.input_edges, storage.host_edges,
                        edge_potentials.size_bytes(), cudaMemcpyHostToDevice,
                        storage.stream),
        "cudaMemcpyAsync edge upload");
  Check(cudaEventRecord(storage.upload_stop, storage.stream),
        "cudaEventRecord upload stop");

  constexpr std::size_t kThreads = 256;
  Check(cudaEventRecord(storage.kernel_start, storage.stream),
        "cudaEventRecord kernel start");
  Params base_params = storage.params;
  base_params.batch = CheckedU32(batch, "batch count");
  base_params.scaled = scaled ? 1 : 0;
  const std::size_t path_batches = base_params.paths_batched ? batch : 1;
  if (scaled) {
    Check(cudaMemsetAsync(storage.node_scales, 0,
                          batch * plan.num_nodes() * sizeof(float),
                          storage.stream),
          "cudaMemsetAsync node scales");
    Check(cudaMemsetAsync(storage.path_scales, 0,
                          path_batches * plan.num_edges() * sizeof(float),
                          storage.stream),
          "cudaMemsetAsync path scales");
    Check(cudaMemsetAsync(storage.branch_scales, 0,
                          batch * plan.num_branches() * sizeof(float),
                          storage.stream),
          "cudaMemsetAsync branch scales");
  }
  initialize_node_data(base_params);
  const std::size_t path_values =
      CheckedProduct({path_batches, plan.num_edges()}, "path matrices");
  if (path_values != 0) {
    InitializePaths<<<Blocks(path_values, kThreads), kThreads, 0,
                      storage.stream>>>(storage.input_edges, storage.paths,
                                        base_params);
  }

  for (const btrc::PrimitiveBatch &primitive_batch : plan.primitive_batches()) {
    Params params = base_params;
    params.operation_offset = primitive_batch.offset;
    params.operation_count = primitive_batch.count;
    const std::size_t operations =
        CheckedProduct({batch, primitive_batch.count}, "primitive CUDA grid");
    const std::uint32_t operation_blocks =
        CheckedU32(operations, "primitive CUDA grid");
    const std::uint32_t serial_blocks = Blocks(operations, kThreads);
    switch (primitive_batch.primitive) {
    case btrc::Primitive::kRake:
      if (states <= 8) {
        RakeSerial<<<serial_blocks, kThreads, 0, storage.stream>>>(
            storage.rakes, storage.nodes, storage.paths, storage.branches,
            storage.node_scales, storage.path_scales, storage.branch_scales,
            params);
      } else {
        Rake<<<operation_blocks, states, 0, storage.stream>>>(
            storage.rakes, storage.nodes, storage.paths, storage.branches,
            storage.node_scales, storage.path_scales, storage.branch_scales,
            params);
      }
      break;
    case btrc::Primitive::kBranchCombination:
      if (states <= 8) {
        CombineBranchesSerial<<<serial_blocks, kThreads, 0, storage.stream>>>(
            storage.combinations, storage.branches, storage.branch_scales,
            params);
      } else {
        CombineBranches<<<operation_blocks, states, 0, storage.stream>>>(
            storage.combinations, storage.branches, storage.branch_scales,
            params);
      }
      break;
    case btrc::Primitive::kBranchAbsorption:
      if (states <= 8) {
        AbsorbBranchesSerial<<<serial_blocks, kThreads, 0, storage.stream>>>(
            storage.absorptions, storage.nodes, storage.branches,
            storage.node_scales, storage.branch_scales, params);
      } else {
        AbsorbBranches<<<operation_blocks, states, 0, storage.stream>>>(
            storage.absorptions, storage.nodes, storage.branches,
            storage.node_scales, storage.branch_scales, params);
      }
      break;
    case btrc::Primitive::kCompression:
      if (states == 4) {
        CompressSerial4<<<serial_blocks, kThreads, 0, storage.stream>>>(
            storage.compressions, storage.nodes, storage.paths,
            storage.node_scales, storage.path_scales, params);
      } else {
        Compress<<<operation_blocks, matrix,
                   (2 * matrix + states) * sizeof(float), storage.stream>>>(
            storage.compressions, storage.nodes, storage.paths,
            storage.node_scales, storage.path_scales, params);
      }
      break;
    }
  }
  FinishRoot<<<CheckedU32(batch, "root CUDA grid"), 1, 0, storage.stream>>>(
      storage.nodes, storage.node_scales, storage.output, base_params);
  Check(cudaGetLastError(), "tree-HMM CUDA kernel launch");
  Check(cudaEventRecord(storage.kernel_stop, storage.stream),
        "cudaEventRecord kernel stop");

  Check(cudaEventRecord(storage.download_start, storage.stream),
        "cudaEventRecord download start");
  Check(cudaMemcpyAsync(storage.host_output, storage.output,
                        batch * sizeof(float), cudaMemcpyDeviceToHost,
                        storage.stream),
        "cudaMemcpyAsync output download");
  Check(cudaEventRecord(storage.download_stop, storage.stream),
        "cudaEventRecord download stop");
  Check(cudaEventSynchronize(storage.download_stop), "tree-HMM CUDA execution");

  float upload_ms = 0.0f;
  float kernel_ms = 0.0f;
  float download_ms = 0.0f;
  Check(cudaEventElapsedTime(&upload_ms, storage.upload_start,
                             storage.upload_stop),
        "cudaEventElapsedTime upload");
  Check(cudaEventElapsedTime(&kernel_ms, storage.kernel_start,
                             storage.kernel_stop),
        "cudaEventElapsedTime kernels");
  Check(cudaEventElapsedTime(&download_ms, storage.download_start,
                             storage.download_stop),
        "cudaEventElapsedTime download");
  const double wall_ms =
      std::chrono::duration<double, std::milli>(Clock::now() - wall_start)
          .count();
  return {{storage.host_output, batch},
          {upload_ms, kernel_ms, download_ms, wall_ms}};
}

template <class StageInputs, class UploadInputs, class InitializeNodeData>
tree_hmm::BatchedMaximumAssignmentView
RunMaximumPrepared(const btrc::Plan &plan, std::size_t states,
                   std::size_t batch, std::span<const float> edge_potentials,
                   Workspace::Impl &storage, StageInputs stage_inputs,
                   UploadInputs upload_inputs,
                   InitializeNodeData initialize_node_data) {
  const auto wall_start = Clock::now();
  if (!storage.bidirectional || storage.plan != &plan ||
      storage.states != states || batch == 0 || batch > storage.batch) {
    throw std::invalid_argument(
        "prepared CUDA MAP inference requires a bidirectional workspace for "
        "this plan, state count, and batch capacity");
  }
  Check(cudaSetDevice(storage.device), "cudaSetDevice");
  const std::size_t matrix = CheckedProduct({states, states}, "state matrix");
  const std::size_t edge_values =
      CheckedProduct({plan.num_edges(), matrix}, "edge inputs");
  if (edge_potentials.size() != edge_values)
    throw std::invalid_argument("CUDA edge input shape is wrong");
  stage_inputs();
  if (edge_potentials.data() != storage.host_edges) {
    std::memcpy(storage.host_edges, edge_potentials.data(),
                edge_potentials.size_bytes());
  }

  Check(cudaEventRecord(storage.upload_start, storage.stream),
        "cudaEventRecord upload start");
  upload_inputs();
  Check(cudaMemcpyAsync(storage.input_edges, storage.host_edges,
                        edge_potentials.size_bytes(), cudaMemcpyHostToDevice,
                        storage.stream),
        "cudaMemcpyAsync edge upload");
  Check(cudaEventRecord(storage.upload_stop, storage.stream),
        "cudaEventRecord upload stop");

  constexpr std::size_t kThreads = 256;
  Check(cudaEventRecord(storage.kernel_start, storage.stream),
        "cudaEventRecord kernel start");
  Params base_params = storage.params;
  base_params.batch = CheckedU32(batch, "batch count");
  base_params.scaled = 0;
  const std::size_t path_batches = base_params.paths_batched ? batch : 1;
  initialize_node_data(base_params);
  const std::size_t path_matrices =
      CheckedProduct({path_batches, plan.num_edges()}, "path matrices");
  if (path_matrices != 0) {
    InitializePaths<<<Blocks(path_matrices, kThreads), kThreads, 0,
                      storage.stream>>>(storage.input_edges, storage.paths,
                                        base_params);
  }
  const std::size_t node_values =
      CheckedProduct({batch, plan.num_nodes(), states}, "MAP node values");
  TakeLogs<<<Blocks(node_values, kThreads), kThreads, 0, storage.stream>>>(
      storage.nodes, node_values);
  const std::size_t path_values = CheckedProduct(
      {path_batches, plan.num_edges(), matrix}, "MAP path values");
  if (path_values != 0) {
    TakeLogs<<<Blocks(path_values, kThreads), kThreads, 0, storage.stream>>>(
        storage.paths, path_values);
  }

  for (const btrc::PrimitiveBatch &primitive_batch : plan.primitive_batches()) {
    Params params = base_params;
    params.operation_offset = primitive_batch.offset;
    params.operation_count = primitive_batch.count;
    const std::size_t operations =
        CheckedProduct({batch, primitive_batch.count}, "MAP primitive grid");
    const std::uint32_t operation_blocks =
        CheckedU32(operations, "MAP primitive grid");
    const std::uint32_t serial_blocks = Blocks(operations, kThreads);
    switch (primitive_batch.primitive) {
    case btrc::Primitive::kRake:
      if (states <= 8) {
        MaximumRakeSerial<<<serial_blocks, kThreads, 0, storage.stream>>>(
            storage.rakes, storage.nodes, storage.paths, storage.branches,
            storage.rake_choices, params);
      } else {
        MaximumRake<<<operation_blocks, states, 0, storage.stream>>>(
            storage.rakes, storage.nodes, storage.paths, storage.branches,
            storage.rake_choices, params);
      }
      break;
    case btrc::Primitive::kBranchCombination:
      if (states <= 8) {
        MaximumCombineBranchesSerial<<<serial_blocks, kThreads, 0,
                                       storage.stream>>>(
            storage.combinations, storage.branches, params);
      } else {
        MaximumCombineBranches<<<operation_blocks, states, 0, storage.stream>>>(
            storage.combinations, storage.branches, params);
      }
      break;
    case btrc::Primitive::kBranchAbsorption:
      if (states <= 8) {
        MaximumAbsorbBranchesSerial<<<serial_blocks, kThreads, 0,
                                      storage.stream>>>(
            storage.absorptions, storage.nodes, storage.branches, params);
      } else {
        MaximumAbsorbBranches<<<operation_blocks, states, 0, storage.stream>>>(
            storage.absorptions, storage.nodes, storage.branches, params);
      }
      break;
    case btrc::Primitive::kCompression:
      if (states == 4) {
        MaximumCompressSerial4<<<serial_blocks, kThreads, 0, storage.stream>>>(
            storage.compressions, storage.nodes, storage.paths,
            storage.compression_choices, params);
      } else {
        MaximumCompress<<<operation_blocks, matrix,
                          (2 * matrix + states) * sizeof(float),
                          storage.stream>>>(
            storage.compressions, storage.nodes, storage.paths,
            storage.compression_choices, params);
      }
      break;
    }
  }

  FinishMaximum<<<Blocks(batch, kThreads), kThreads, 0, storage.stream>>>(
      storage.nodes, storage.output, storage.assignments, base_params);
  for (std::size_t index = plan.primitive_batches().size(); index-- > 0;) {
    const btrc::PrimitiveBatch &primitive_batch =
        plan.primitive_batches()[index];
    Params params = base_params;
    params.operation_offset = primitive_batch.offset;
    params.operation_count = primitive_batch.count;
    const std::size_t operations =
        CheckedProduct({batch, primitive_batch.count}, "MAP expansion grid");
    switch (primitive_batch.primitive) {
    case btrc::Primitive::kRake:
      ExpandMaximumRakes<<<Blocks(operations, kThreads), kThreads, 0,
                           storage.stream>>>(
          storage.rakes, storage.rake_choices, storage.assignments, params);
      break;
    case btrc::Primitive::kCompression:
      ExpandMaximumCompressions<<<Blocks(operations, kThreads), kThreads, 0,
                                  storage.stream>>>(
          storage.compressions, storage.compression_choices,
          storage.assignments, params);
      break;
    case btrc::Primitive::kBranchCombination:
    case btrc::Primitive::kBranchAbsorption:
      break;
    }
  }
  Check(cudaGetLastError(), "tree-HMM CUDA MAP kernel launch");
  Check(cudaEventRecord(storage.kernel_stop, storage.stream),
        "cudaEventRecord kernel stop");

  Check(cudaEventRecord(storage.download_start, storage.stream),
        "cudaEventRecord download start");
  Check(cudaMemcpyAsync(storage.host_output, storage.output,
                        batch * sizeof(float), cudaMemcpyDeviceToHost,
                        storage.stream),
        "cudaMemcpyAsync MAP-weight download");
  const std::size_t assignment_count =
      CheckedProduct({batch, plan.num_nodes()}, "MAP assignment download");
  Check(cudaMemcpyAsync(storage.host_assignments, storage.assignments,
                        assignment_count * sizeof(std::uint32_t),
                        cudaMemcpyDeviceToHost, storage.stream),
        "cudaMemcpyAsync MAP-assignment download");
  Check(cudaEventRecord(storage.download_stop, storage.stream),
        "cudaEventRecord download stop");
  Check(cudaEventSynchronize(storage.download_stop),
        "tree-HMM CUDA MAP execution");

  for (std::size_t batch_index = 0; batch_index < batch; ++batch_index) {
    if (!std::isfinite(storage.host_output[batch_index])) {
      throw std::domain_error("the tree HMM has no positive-weight assignment");
    }
  }
  float upload_ms = 0.0f;
  float kernel_ms = 0.0f;
  float download_ms = 0.0f;
  Check(cudaEventElapsedTime(&upload_ms, storage.upload_start,
                             storage.upload_stop),
        "cudaEventElapsedTime MAP upload");
  Check(cudaEventElapsedTime(&kernel_ms, storage.kernel_start,
                             storage.kernel_stop),
        "cudaEventElapsedTime MAP kernels");
  Check(cudaEventElapsedTime(&download_ms, storage.download_start,
                             storage.download_stop),
        "cudaEventElapsedTime MAP download");
  const double wall_ms =
      std::chrono::duration<double, std::milli>(Clock::now() - wall_start)
          .count();
  return {{storage.host_output, batch},
          {storage.host_assignments, assignment_count},
          {upload_ms, kernel_ms, download_ms, wall_ms}};
}

tree_hmm::PartitionView Run(tree_hmm::BatchedModelView model,
                            Workspace::Impl &storage, bool scaled) {
  if (storage.categorical)
    throw std::invalid_argument(
        "dense CUDA inference cannot use a categorical workspace");
  const std::size_t node_values = CheckedProduct(
      {model.batch, model.plan.num_nodes(), model.states}, "node inputs");
  if (model.node_potentials.size() != node_values)
    throw std::invalid_argument("CUDA node input shape is wrong");
  return RunPrepared(
      model.plan, model.states, model.batch, model.edge_potentials, storage,
      scaled,
      [&] {
        if (model.node_potentials.data() != storage.host_nodes) {
          std::memcpy(storage.host_nodes, model.node_potentials.data(),
                      model.node_potentials.size_bytes());
        }
      },
      [&] {
        Check(cudaMemcpyAsync(storage.input_nodes, storage.host_nodes,
                              model.node_potentials.size_bytes(),
                              cudaMemcpyHostToDevice, storage.stream),
              "cudaMemcpyAsync node upload");
      },
      [&](const Params &params) {
        const dim3 threads(kTransposeTile, kTransposeRows);
        const dim3 blocks((model.plan.num_nodes() + kTransposeTile - 1) /
                              kTransposeTile,
                          (model.batch + kTransposeTile - 1) / kTransposeTile);
        InitializeNodes<<<blocks, threads, 0, storage.stream>>>(
            storage.input_nodes, storage.nodes, params);
      });
}

tree_hmm::PartitionView Run(tree_hmm::BatchedCategoricalModelView model,
                            Workspace::Impl &storage, bool scaled) {
  if (!storage.categorical || model.categories != storage.categories ||
      model.observation_nodes.size() != storage.observation_nodes.size() ||
      !std::equal(model.observation_nodes.begin(),
                  model.observation_nodes.end(),
                  storage.observation_nodes.begin())) {
    throw std::invalid_argument(
        "categorical CUDA model does not match the reserved workspace");
  }
  const std::size_t observation_values =
      CheckedProduct({model.batch, model.observation_nodes.size()},
                     "categorical observations");
  const std::size_t emission_values =
      CheckedProduct({model.categories, model.states}, "categorical emissions");
  if (model.observations.size() != observation_values ||
      model.root_potential.size() != model.states ||
      model.emission_potentials.size() != emission_values) {
    throw std::invalid_argument("CUDA categorical input shapes are wrong");
  }
  return RunPrepared(
      model.plan, model.states, model.batch, model.edge_potentials, storage,
      scaled,
      [&] {
        if (model.observations.data() != storage.host_observations) {
          std::memcpy(storage.host_observations, model.observations.data(),
                      model.observations.size_bytes());
        }
        if (model.root_potential.data() != storage.host_root_potential) {
          std::memcpy(storage.host_root_potential, model.root_potential.data(),
                      model.root_potential.size_bytes());
        }
        if (model.emission_potentials.data() !=
            storage.host_emission_potentials) {
          std::memcpy(storage.host_emission_potentials,
                      model.emission_potentials.data(),
                      model.emission_potentials.size_bytes());
        }
      },
      [&] {
        Check(cudaMemcpyAsync(storage.input_observations,
                              storage.host_observations,
                              model.observations.size_bytes(),
                              cudaMemcpyHostToDevice, storage.stream),
              "cudaMemcpyAsync categorical observation upload");
        Check(cudaMemcpyAsync(storage.input_root_potential,
                              storage.host_root_potential,
                              model.root_potential.size_bytes(),
                              cudaMemcpyHostToDevice, storage.stream),
              "cudaMemcpyAsync root-potential upload");
        Check(cudaMemcpyAsync(storage.input_emission_potentials,
                              storage.host_emission_potentials,
                              model.emission_potentials.size_bytes(),
                              cudaMemcpyHostToDevice, storage.stream),
              "cudaMemcpyAsync emission-potential upload");
      },
      [&](const Params &params) {
        constexpr std::size_t kThreads = 256;
        const std::size_t node_batches =
            CheckedProduct({model.plan.num_nodes(), model.batch},
                           "categorical node initialization");
        InitializeCategoricalBase<<<Blocks(node_batches, kThreads), kThreads, 0,
                                    storage.stream>>>(
            storage.input_root_potential,
            storage.categorical_observation_index_by_node, storage.nodes,
            params);
        if (!model.observation_nodes.empty()) {
          const dim3 threads(kTransposeTile, kTransposeRows);
          const dim3 blocks(
              (model.observation_nodes.size() + kTransposeTile - 1) /
                  kTransposeTile,
              (model.batch + kTransposeTile - 1) / kTransposeTile);
          ApplyCategoricalObservations<<<blocks, threads, 0, storage.stream>>>(
              storage.input_observations, storage.categorical_observation_nodes,
              storage.input_root_potential, storage.input_emission_potentials,
              storage.nodes, params,
              CheckedU32(model.observation_nodes.size(),
                         "categorical observation count"),
              CheckedU32(model.categories, "category count"));
        }
      });
}

tree_hmm::BatchedMaximumAssignmentView
RunMaximum(tree_hmm::BatchedModelView model, Workspace::Impl &storage) {
  if (storage.categorical)
    throw std::invalid_argument(
        "dense CUDA inference cannot use a categorical workspace");
  const std::size_t node_values = CheckedProduct(
      {model.batch, model.plan.num_nodes(), model.states}, "node inputs");
  if (model.node_potentials.size() != node_values)
    throw std::invalid_argument("CUDA node input shape is wrong");
  return RunMaximumPrepared(
      model.plan, model.states, model.batch, model.edge_potentials, storage,
      [&] {
        if (model.node_potentials.data() != storage.host_nodes) {
          std::memcpy(storage.host_nodes, model.node_potentials.data(),
                      model.node_potentials.size_bytes());
        }
      },
      [&] {
        Check(cudaMemcpyAsync(storage.input_nodes, storage.host_nodes,
                              model.node_potentials.size_bytes(),
                              cudaMemcpyHostToDevice, storage.stream),
              "cudaMemcpyAsync node upload");
      },
      [&](const Params &params) {
        const dim3 threads(kTransposeTile, kTransposeRows);
        const dim3 blocks((model.plan.num_nodes() + kTransposeTile - 1) /
                              kTransposeTile,
                          (model.batch + kTransposeTile - 1) / kTransposeTile);
        InitializeNodes<<<blocks, threads, 0, storage.stream>>>(
            storage.input_nodes, storage.nodes, params);
      });
}

tree_hmm::BatchedMaximumAssignmentView
RunMaximum(tree_hmm::BatchedCategoricalModelView model,
           Workspace::Impl &storage) {
  if (!storage.categorical || model.categories != storage.categories ||
      model.observation_nodes.size() != storage.observation_nodes.size() ||
      !std::equal(model.observation_nodes.begin(),
                  model.observation_nodes.end(),
                  storage.observation_nodes.begin())) {
    throw std::invalid_argument(
        "categorical CUDA model does not match the reserved workspace");
  }
  const std::size_t observation_values =
      CheckedProduct({model.batch, model.observation_nodes.size()},
                     "categorical observations");
  const std::size_t emission_values =
      CheckedProduct({model.categories, model.states}, "categorical emissions");
  if (model.observations.size() != observation_values ||
      model.root_potential.size() != model.states ||
      model.emission_potentials.size() != emission_values) {
    throw std::invalid_argument("CUDA categorical input shapes are wrong");
  }
  return RunMaximumPrepared(
      model.plan, model.states, model.batch, model.edge_potentials, storage,
      [&] {
        if (model.observations.data() != storage.host_observations) {
          std::memcpy(storage.host_observations, model.observations.data(),
                      model.observations.size_bytes());
        }
        if (model.root_potential.data() != storage.host_root_potential) {
          std::memcpy(storage.host_root_potential, model.root_potential.data(),
                      model.root_potential.size_bytes());
        }
        if (model.emission_potentials.data() !=
            storage.host_emission_potentials) {
          std::memcpy(storage.host_emission_potentials,
                      model.emission_potentials.data(),
                      model.emission_potentials.size_bytes());
        }
      },
      [&] {
        Check(cudaMemcpyAsync(storage.input_observations,
                              storage.host_observations,
                              model.observations.size_bytes(),
                              cudaMemcpyHostToDevice, storage.stream),
              "cudaMemcpyAsync categorical observation upload");
        Check(cudaMemcpyAsync(storage.input_root_potential,
                              storage.host_root_potential,
                              model.root_potential.size_bytes(),
                              cudaMemcpyHostToDevice, storage.stream),
              "cudaMemcpyAsync root-potential upload");
        Check(cudaMemcpyAsync(storage.input_emission_potentials,
                              storage.host_emission_potentials,
                              model.emission_potentials.size_bytes(),
                              cudaMemcpyHostToDevice, storage.stream),
              "cudaMemcpyAsync emission-potential upload");
      },
      [&](const Params &params) {
        constexpr std::size_t kThreads = 256;
        const std::size_t node_batches =
            CheckedProduct({model.plan.num_nodes(), model.batch},
                           "categorical node initialization");
        InitializeCategoricalBase<<<Blocks(node_batches, kThreads), kThreads, 0,
                                    storage.stream>>>(
            storage.input_root_potential,
            storage.categorical_observation_index_by_node, storage.nodes,
            params);
        if (!model.observation_nodes.empty()) {
          const dim3 threads(kTransposeTile, kTransposeRows);
          const dim3 blocks(
              (model.observation_nodes.size() + kTransposeTile - 1) /
                  kTransposeTile,
              (model.batch + kTransposeTile - 1) / kTransposeTile);
          ApplyCategoricalObservations<<<blocks, threads, 0, storage.stream>>>(
              storage.input_observations, storage.categorical_observation_nodes,
              storage.input_root_potential, storage.input_emission_potentials,
              storage.nodes, params,
              CheckedU32(model.observation_nodes.size(),
                         "categorical observation count"),
              CheckedU32(model.categories, "category count"));
        }
      });
}

} // namespace

tree_hmm::PartitionView
PartitionFunctionPrepared(tree_hmm::BatchedModelView model,
                          Workspace &workspace) {
  return Run(model, *workspace.impl_, false);
}

tree_hmm::PartitionView
LogPartitionFunctionPrepared(tree_hmm::BatchedModelView model,
                             Workspace &workspace) {
  return Run(model, *workspace.impl_, true);
}

tree_hmm::PartitionView
PartitionFunctionPrepared(tree_hmm::BatchedCategoricalModelView model,
                          Workspace &workspace) {
  return Run(model, *workspace.impl_, false);
}

tree_hmm::PartitionView
LogPartitionFunctionPrepared(tree_hmm::BatchedCategoricalModelView model,
                             Workspace &workspace) {
  return Run(model, *workspace.impl_, true);
}

tree_hmm::BatchedMaximumAssignmentView
MaximumAPosterioriPrepared(tree_hmm::BatchedModelView model,
                           Workspace &workspace) {
  return RunMaximum(model, *workspace.impl_);
}

tree_hmm::BatchedMaximumAssignmentView
MaximumAPosterioriPrepared(tree_hmm::BatchedCategoricalModelView model,
                           Workspace &workspace) {
  return RunMaximum(model, *workspace.impl_);
}

} // namespace tree_hmm::cuda
