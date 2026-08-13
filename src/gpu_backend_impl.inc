#if !defined(TREE_HMM_GPU_BACKEND_NAMESPACE)
#error "gpu_backend_impl.inc requires a backend wrapper"
#endif

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

#include "src/accelerator_path_storage.h"
#include "src/gpu_device_algebra.h"

namespace tree_hmm::TREE_HMM_GPU_BACKEND_NAMESPACE {
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
  std::uint32_t mutable_paths;
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
                     CheckedProduct({count, sizeof(Value)}, "accelerator buffer"), 1)),
      "GPU runtime Malloc");
}

template <class Value> void HostAllocate(Value *&pointer, std::size_t count) {
  Check(
      cudaMallocHost(
          reinterpret_cast<void **>(&pointer),
          std::max<std::size_t>(
              CheckedProduct({count, sizeof(Value)}, "pinned host buffer"), 1)),
      "GPU runtime MallocHost");
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
          "GPU runtime MemcpyAsync topology upload");
  }
}

template <class Value, class Allocator>
void Upload(Value *destination, const std::vector<Value, Allocator> &source,
            cudaStream_t stream) {
  Upload(destination, std::span<const Value>(source), stream);
}

__device__ std::size_t NodeIndex(const Params &params, std::size_t batch,
                                 std::size_t node, std::size_t state) {
  return (node * params.batch + batch) * params.states + state;
}

__device__ std::size_t PathIndex(const Params &params,
                                 const btrc::Index *mutable_slot_by_edge,
                                 std::size_t batch, std::size_t edge,
                                 std::size_t parent_state,
                                 std::size_t child_state) {
  const btrc::Index slot = mutable_slot_by_edge[edge];
  const std::size_t path =
      slot == tree_hmm::internal::kImmutablePath
          ? edge
          : params.edges + static_cast<std::size_t>(slot) * params.batch +
                batch;
  return (path * params.states + parent_state) * params.states + child_state;
}

__device__ std::size_t BranchIndex(const Params &params, std::size_t batch,
                                   std::size_t branch, std::size_t state) {
  return (branch * params.batch + batch) * params.states + state;
}

__device__ std::size_t NodeScaleIndex(const Params &params, std::size_t batch,
                                      std::size_t node) {
  return node * params.batch + batch;
}

__device__ std::size_t PathScaleIndex(const Params &params,
                                      const btrc::Index *mutable_slot_by_edge,
                                      std::size_t batch, std::size_t edge) {
  const btrc::Index slot = mutable_slot_by_edge[edge];
  return slot == tree_hmm::internal::kImmutablePath
             ? edge
             : params.edges + static_cast<std::size_t>(slot) * params.batch +
                   batch;
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

__device__ std::size_t RakePathTapeIndex(const Params &params,
                                         std::size_t batch, std::size_t branch,
                                         std::size_t entry) {
  const std::size_t matrix =
      static_cast<std::size_t>(params.states) * params.states;
  return (branch * params.batch + batch) * matrix + entry;
}

__device__ std::size_t RakeLeafTapeIndex(const Params &params,
                                         std::size_t batch, std::size_t branch,
                                         std::size_t state) {
  return (branch * params.batch + batch) * params.states + state;
}

__device__ std::size_t CompressionMatrixTapeIndex(const Params &params,
                                                  std::size_t batch,
                                                  std::size_t tape,
                                                  std::size_t entry) {
  const std::size_t matrix =
      static_cast<std::size_t>(params.states) * params.states;
  return (tape * params.batch + batch) * matrix + entry;
}

__device__ std::size_t CompressionVectorTapeIndex(const Params &params,
                                                  std::size_t batch,
                                                  std::size_t tape,
                                                  std::size_t state) {
  return (tape * params.batch + batch) * params.states + state;
}

__device__ std::size_t NodeMarginalIndex(const Params &params,
                                         std::size_t batch, std::size_t node,
                                         std::size_t state) {
  return (batch * params.nodes + node) * params.states + state;
}

__device__ std::size_t EdgeMarginalIndex(const Params &params,
                                         std::size_t batch, std::size_t edge,
                                         std::size_t parent_state,
                                         std::size_t child_state) {
  return ((batch * params.edges + edge) * params.states + parent_state) *
             params.states +
         child_state;
}

__device__ std::uint32_t
SampleContiguous(const Scalar *weights, std::uint32_t states, Scalar uniform) {
  Scalar total = 0.0f;
  for (std::uint32_t state = 0; state < states; ++state)
    total += weights[state];
  const Scalar threshold = uniform * total;
  Scalar cumulative = 0.0f;
  std::uint32_t fallback = 0;
  for (std::uint32_t state = 0; state < states; ++state) {
    const Scalar value = weights[state];
    if (value > 0.0f)
      fallback = state;
    cumulative += value;
    if (threshold < cumulative)
      return state;
  }
  return fallback;
}

__device__ Scalar LogProduct(Scalar left, Scalar right) {
  return left > 0.0f && right > 0.0f ? log(left) + log(right) : -INFINITY;
}

__device__ Scalar LogProduct(Scalar left, Scalar middle, Scalar right) {
  return left > 0.0f && middle > 0.0f && right > 0.0f
             ? log(left) + log(middle) + log(right)
             : -INFINITY;
}

__device__ bool Finite(Scalar value) {
  return value > -INFINITY && value < INFINITY;
}

struct SmallStateWorkItem {
  std::uint32_t operation;
  std::uint32_t batch;
};

// Small-state kernels assign one complete algebra operation to each thread for
// every Scalar type. Keeping operations and batch items on separate grid axes
// avoids recovering both coordinates with integer division in every primitive.
__device__ SmallStateWorkItem SmallStateWork() {
  return {static_cast<std::uint32_t>(blockIdx.x * blockDim.y + threadIdx.y),
          static_cast<std::uint32_t>(
              (blockIdx.z * gridDim.y + blockIdx.y) * blockDim.x +
              threadIdx.x)};
}

// This execution threshold depends only on state count; the same kernels and
// policy are compiled for every Scalar type and both CUDA-compatible runtimes.
constexpr std::size_t kSmallStateLimit = 8;

// Transpose the public [batch, node, state] input into the internal
// [node, batch, state] layout without constraining the state count.
constexpr std::size_t kTransposeTile = 32;
constexpr std::size_t kTransposeRows = 8;

__global__ void InitializeNodes(const Scalar *input, Scalar *nodes,
                                Params params) {
  __shared__ Scalar tile[kTransposeTile][kTransposeTile + 1];
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
InitializeCategoricalBase(const Scalar *root_potential,
                          const btrc::Index *observation_index_by_node,
                          Scalar *nodes, Params params) {
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
    const Scalar *root_potential, const Scalar *emission_potentials,
    Scalar *nodes, Params params, std::uint32_t observation_count,
    std::uint32_t categories) {
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
      const Scalar emission =
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

__global__ void InitializePaths(const Scalar *input,
                                const btrc::Index *mutable_path_edges,
                                Scalar *paths, Params params) {
  const std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  const std::size_t count =
      static_cast<std::size_t>(params.edges) +
      static_cast<std::size_t>(params.mutable_paths) * params.batch;
  if (index >= count)
    return;
  const bool immutable = index < params.edges;
  const std::size_t mutable_index = immutable ? 0 : index - params.edges;
  const std::size_t mutable_slot = immutable ? 0 : mutable_index / params.batch;
  const std::size_t edge = immutable ? index : mutable_path_edges[mutable_slot];
  const std::size_t matrix =
      static_cast<std::size_t>(params.states) * params.states;
  const std::size_t output = index * matrix;
  const std::size_t source = edge * matrix;
  for (std::size_t entry = 0; entry < matrix; ++entry)
    paths[output + entry] = input[source + entry];
}

__global__ void TakeLogs(Scalar *values, std::size_t count) {
  const std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < count)
    values[index] = values[index] > 0.0f ? log(values[index]) : -INFINITY;
}

__global__ void LogRake(const btrc::Rake *operations, const Scalar *nodes,
                        const Scalar *paths,
                        const btrc::Index *mutable_path_slots, Scalar *branches,
                        Scalar *path_tape, Scalar *leaf_tape,
                        Scalar *message_tape, Params params) {
  const std::size_t parent_state = threadIdx.x;
  const std::size_t batch = blockIdx.x % params.batch;
  const std::size_t operation_in_batch = blockIdx.x / params.batch;
  if (parent_state >= params.states || batch >= params.batch)
    return;
  const btrc::Rake operation =
      operations[params.operation_offset + operation_in_batch];
  const std::size_t node_base = NodeIndex(params, batch, operation.leaf, 0);
  const std::size_t path_base =
      PathIndex(params, mutable_path_slots, batch, operation.edge, 0, 0);
  leaf_tape[RakeLeafTapeIndex(params, batch, operation.branch, parent_state)] =
      nodes[node_base + parent_state];
  for (std::size_t child_state = 0; child_state < params.states;
       ++child_state) {
    path_tape[RakePathTapeIndex(params, batch, operation.branch,
                                parent_state * params.states + child_state)] =
        paths[path_base + parent_state * params.states + child_state];
  }

  Scalar maximum = -INFINITY;
  for (std::size_t child_state = 0; child_state < params.states;
       ++child_state) {
    maximum = fmax(
        maximum, paths[path_base + parent_state * params.states + child_state] +
                     nodes[node_base + child_state]);
  }
  Scalar message = maximum;
  if (Finite(maximum)) {
    Scalar total = 0.0f;
    for (std::size_t child_state = 0; child_state < params.states;
         ++child_state) {
      total +=
          exp(paths[path_base + parent_state * params.states + child_state] +
              nodes[node_base + child_state] - maximum);
    }
    message += log(total);
  }
  branches[BranchIndex(params, batch, operation.branch, parent_state)] =
      message;
  message_tape[RakeLeafTapeIndex(params, batch, operation.branch,
                                 parent_state)] = message;
}

__global__ void LogCompress(const btrc::Compression *operations,
                            const Scalar *nodes, Scalar *paths,
                            const btrc::Index *mutable_path_slots,
                            Scalar *left_tape, Scalar *middle_tape,
                            Scalar *right_tape, Scalar *output_tape,
                            Params params) {
  extern __shared__ Scalar storage[];
  const std::size_t matrix =
      static_cast<std::size_t>(params.states) * params.states;
  Scalar *left = storage;
  Scalar *right = left + matrix;
  Scalar *middle = right + matrix;
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
    const std::size_t tape_index =
        CompressionMatrixTapeIndex(params, batch, operation.tape, entry);
    left[entry] =
        paths[PathIndex(params, mutable_path_slots, batch, operation.left_edge,
                        parent_state, child_state)];
    right[entry] =
        paths[PathIndex(params, mutable_path_slots, batch, operation.right_edge,
                        parent_state, child_state)];
    left_tape[tape_index] = left[entry];
    right_tape[tape_index] = right[entry];
  }
  if (entry < params.states) {
    middle[entry] = nodes[NodeIndex(params, batch, operation.middle, entry)];
    middle_tape[CompressionVectorTapeIndex(params, batch, operation.tape,
                                           entry)] = middle[entry];
  }
  __syncthreads();
  if (entry >= matrix)
    return;

  const std::size_t parent_state = entry / params.states;
  const std::size_t child_state = entry % params.states;
  Scalar maximum = -INFINITY;
  for (std::size_t middle_state = 0; middle_state < params.states;
       ++middle_state) {
    maximum =
        fmax(maximum, left[parent_state * params.states + middle_state] +
                          middle[middle_state] +
                          right[middle_state * params.states + child_state]);
  }
  Scalar output = maximum;
  if (Finite(maximum)) {
    Scalar total = 0.0f;
    for (std::size_t middle_state = 0; middle_state < params.states;
         ++middle_state) {
      total += exp(left[parent_state * params.states + middle_state] +
                   middle[middle_state] +
                   right[middle_state * params.states + child_state] - maximum);
    }
    output += log(total);
  }
  paths[PathIndex(params, mutable_path_slots, batch, operation.left_edge,
                  parent_state, child_state)] = output;
  output_tape[CompressionMatrixTapeIndex(params, batch, operation.tape,
                                         entry)] = output;
}

__global__ void FinishLogRootAndSeedMarginals(const Scalar *nodes,
                                              Scalar *log_partitions,
                                              Scalar *node_marginals,
                                              Params params) {
  const std::size_t batch = blockIdx.x * blockDim.x + threadIdx.x;
  if (batch >= params.batch)
    return;
  const std::size_t root_base = NodeIndex(params, batch, params.root, 0);
  Scalar maximum = -INFINITY;
  for (std::size_t state = 0; state < params.states; ++state)
    maximum = fmax(maximum, nodes[root_base + state]);
  Scalar log_partition = maximum;
  if (Finite(maximum)) {
    Scalar total = 0.0f;
    for (std::size_t state = 0; state < params.states; ++state)
      total += exp(nodes[root_base + state] - maximum);
    log_partition += log(total);
  }
  log_partitions[batch] = log_partition;
  for (std::size_t state = 0; state < params.states; ++state) {
    node_marginals[NodeMarginalIndex(params, batch, params.root, state)] =
        Finite(log_partition) ? exp(nodes[root_base + state] - log_partition)
                              : 0.0f;
  }
}

__global__ void ReverseLogCompressions(const btrc::Compression *operations,
                                       const Scalar *left_tape,
                                       const Scalar *middle_tape,
                                       const Scalar *right_tape,
                                       const Scalar *output_tape,
                                       Scalar *node_marginals,
                                       Scalar *edge_marginals, Params params) {
  extern __shared__ Scalar output_adjoints[];
  const std::size_t matrix =
      static_cast<std::size_t>(params.states) * params.states;
  const std::size_t entry = threadIdx.x;
  const std::size_t batch = blockIdx.x % params.batch;
  const std::size_t operation_in_batch = blockIdx.x / params.batch;
  if (batch >= params.batch)
    return;
  const btrc::Compression operation =
      operations[params.operation_offset + operation_in_batch];
  if (entry < matrix) {
    const std::size_t parent_state = entry / params.states;
    const std::size_t child_state = entry % params.states;
    const std::size_t output_index = EdgeMarginalIndex(
        params, batch, operation.left_edge, parent_state, child_state);
    output_adjoints[entry] = edge_marginals[output_index];
    edge_marginals[output_index] = 0.0f;
  }
  __syncthreads();
  if (entry >= matrix)
    return;

  const std::size_t first_state = entry / params.states;
  const std::size_t second_state = entry % params.states;
  const std::size_t left_tape_index =
      CompressionMatrixTapeIndex(params, batch, operation.tape, entry);
  const Scalar left = left_tape[left_tape_index];
  Scalar left_adjoint = 0.0f;
  for (std::size_t child_state = 0; child_state < params.states;
       ++child_state) {
    const std::size_t output_entry = first_state * params.states + child_state;
    const Scalar output = output_tape[CompressionMatrixTapeIndex(
        params, batch, operation.tape, output_entry)];
    const Scalar term = left +
                        middle_tape[CompressionVectorTapeIndex(
                            params, batch, operation.tape, second_state)] +
                        right_tape[CompressionMatrixTapeIndex(
                            params, batch, operation.tape,
                            second_state * params.states + child_state)];
    if (output_adjoints[output_entry] != 0.0f && Finite(term) &&
        Finite(output)) {
      left_adjoint += output_adjoints[output_entry] * exp(term - output);
    }
  }
  edge_marginals[EdgeMarginalIndex(params, batch, operation.left_edge,
                                   first_state, second_state)] = left_adjoint;

  const Scalar right = right_tape[left_tape_index];
  Scalar right_adjoint = 0.0f;
  for (std::size_t parent_state = 0; parent_state < params.states;
       ++parent_state) {
    const std::size_t output_entry =
        parent_state * params.states + second_state;
    const Scalar output = output_tape[CompressionMatrixTapeIndex(
        params, batch, operation.tape, output_entry)];
    const Scalar term = left_tape[CompressionMatrixTapeIndex(
                            params, batch, operation.tape,
                            parent_state * params.states + first_state)] +
                        middle_tape[CompressionVectorTapeIndex(
                            params, batch, operation.tape, first_state)] +
                        right;
    if (output_adjoints[output_entry] != 0.0f && Finite(term) &&
        Finite(output)) {
      right_adjoint += output_adjoints[output_entry] * exp(term - output);
    }
  }
  edge_marginals[EdgeMarginalIndex(params, batch, operation.right_edge,
                                   first_state, second_state)] += right_adjoint;

  if (entry < params.states) {
    const std::size_t middle_state = entry;
    const Scalar middle = middle_tape[CompressionVectorTapeIndex(
        params, batch, operation.tape, middle_state)];
    Scalar middle_adjoint = 0.0f;
    for (std::size_t parent_state = 0; parent_state < params.states;
         ++parent_state) {
      for (std::size_t child_state = 0; child_state < params.states;
           ++child_state) {
        const std::size_t output_entry =
            parent_state * params.states + child_state;
        const Scalar output = output_tape[CompressionMatrixTapeIndex(
            params, batch, operation.tape, output_entry)];
        const Scalar term = left_tape[CompressionMatrixTapeIndex(
                                params, batch, operation.tape,
                                parent_state * params.states + middle_state)] +
                            middle +
                            right_tape[CompressionMatrixTapeIndex(
                                params, batch, operation.tape,
                                middle_state * params.states + child_state)];
        if (output_adjoints[output_entry] != 0.0f && Finite(term) &&
            Finite(output)) {
          middle_adjoint += output_adjoints[output_entry] * exp(term - output);
        }
      }
    }
    node_marginals[NodeMarginalIndex(params, batch, operation.middle,
                                     middle_state)] += middle_adjoint;
  }
}

__global__ void ReverseLogAbsorptions(const btrc::BranchAbsorption *operations,
                                      const Scalar *node_marginals,
                                      Scalar *branch_marginals, Params params) {
  const std::size_t state = threadIdx.x;
  const std::size_t batch = blockIdx.x % params.batch;
  const std::size_t operation_in_batch = blockIdx.x / params.batch;
  if (state >= params.states || batch >= params.batch)
    return;
  const btrc::BranchAbsorption operation =
      operations[params.operation_offset + operation_in_batch];
  branch_marginals[BranchIndex(params, batch, operation.branch, state)] +=
      node_marginals[NodeMarginalIndex(params, batch, operation.parent, state)];
}

__global__ void
ReverseLogCombinations(const btrc::BranchCombination *operations,
                       Scalar *branch_marginals, Params params) {
  const std::size_t state = threadIdx.x;
  const std::size_t batch = blockIdx.x % params.batch;
  const std::size_t operation_in_batch = blockIdx.x / params.batch;
  if (state >= params.states || batch >= params.batch)
    return;
  const btrc::BranchCombination operation =
      operations[params.operation_offset + operation_in_batch];
  branch_marginals[BranchIndex(params, batch, operation.source, state)] +=
      branch_marginals[BranchIndex(params, batch, operation.destination,
                                   state)];
}

__global__ void
ReverseLogRakes(const btrc::Rake *operations, const Scalar *path_tape,
                const Scalar *leaf_tape, const Scalar *message_tape,
                const Scalar *branch_marginals, Scalar *node_marginals,
                Scalar *edge_marginals, Params params) {
  const std::size_t matrix =
      static_cast<std::size_t>(params.states) * params.states;
  const std::size_t entry = threadIdx.x;
  const std::size_t batch = blockIdx.x % params.batch;
  const std::size_t operation_in_batch = blockIdx.x / params.batch;
  if (batch >= params.batch || entry >= matrix)
    return;
  const btrc::Rake operation =
      operations[params.operation_offset + operation_in_batch];
  const std::size_t parent_state = entry / params.states;
  const std::size_t child_state = entry % params.states;
  const Scalar message = message_tape[RakeLeafTapeIndex(
      params, batch, operation.branch, parent_state)];
  const Scalar term =
      path_tape[RakePathTapeIndex(params, batch, operation.branch, entry)] +
      leaf_tape[RakeLeafTapeIndex(params, batch, operation.branch,
                                  child_state)];
  Scalar contribution = 0.0f;
  if (Finite(term) && Finite(message)) {
    contribution = branch_marginals[BranchIndex(params, batch, operation.branch,
                                                parent_state)] *
                   exp(term - message);
  }
  edge_marginals[EdgeMarginalIndex(params, batch, operation.edge, parent_state,
                                   child_state)] += contribution;

  if (entry < params.states) {
    const std::size_t leaf_state = entry;
    Scalar leaf_adjoint = 0.0f;
    for (std::size_t rake_parent_state = 0; rake_parent_state < params.states;
         ++rake_parent_state) {
      const Scalar rake_message = message_tape[RakeLeafTapeIndex(
          params, batch, operation.branch, rake_parent_state)];
      const Scalar rake_term =
          path_tape[RakePathTapeIndex(params, batch, operation.branch,
                                      rake_parent_state * params.states +
                                          leaf_state)] +
          leaf_tape[RakeLeafTapeIndex(params, batch, operation.branch,
                                      leaf_state)];
      if (Finite(rake_term) && Finite(rake_message)) {
        leaf_adjoint +=
            branch_marginals[BranchIndex(params, batch, operation.branch,
                                         rake_parent_state)] *
            exp(rake_term - rake_message);
      }
    }
    node_marginals[NodeMarginalIndex(params, batch, operation.leaf,
                                     leaf_state)] += leaf_adjoint;
  }
}

__global__ void SaveRakeTapes(const btrc::Rake *operations, const Scalar *nodes,
                              const Scalar *paths,
                              const btrc::Index *mutable_path_slots,
                              Scalar *path_tape, Scalar *leaf_tape,
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
  const std::size_t matrix =
      static_cast<std::size_t>(params.states) * params.states;
  const std::size_t path_base =
      PathIndex(params, mutable_path_slots, batch, operation.edge, 0, 0);
  const std::size_t node_base = NodeIndex(params, batch, operation.leaf, 0);
  for (std::size_t entry = 0; entry < matrix; ++entry) {
    path_tape[RakePathTapeIndex(params, batch, operation.branch, entry)] =
        paths[path_base + entry];
  }
  for (std::size_t state = 0; state < params.states; ++state) {
    leaf_tape[RakeLeafTapeIndex(params, batch, operation.branch, state)] =
        nodes[node_base + state];
  }
}

__global__ void SaveCompressionTapes(const btrc::Compression *operations,
                                     const Scalar *nodes, const Scalar *paths,
                                     const btrc::Index *mutable_path_slots,
                                     Scalar *left_tape, Scalar *middle_tape,
                                     Scalar *right_tape, Params params) {
  const std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  const std::size_t count =
      static_cast<std::size_t>(params.batch) * params.operation_count;
  if (index >= count)
    return;
  const std::size_t batch = index % params.batch;
  const std::size_t operation_in_batch = index / params.batch;
  const btrc::Compression operation =
      operations[params.operation_offset + operation_in_batch];
  const std::size_t matrix =
      static_cast<std::size_t>(params.states) * params.states;
  const std::size_t left_base =
      PathIndex(params, mutable_path_slots, batch, operation.left_edge, 0, 0);
  const std::size_t right_base =
      PathIndex(params, mutable_path_slots, batch, operation.right_edge, 0, 0);
  const std::size_t middle_base = NodeIndex(params, batch, operation.middle, 0);
  for (std::size_t entry = 0; entry < matrix; ++entry) {
    const std::size_t tape_index =
        CompressionMatrixTapeIndex(params, batch, operation.tape, entry);
    left_tape[tape_index] = paths[left_base + entry];
    right_tape[tape_index] = paths[right_base + entry];
  }
  for (std::size_t state = 0; state < params.states; ++state) {
    middle_tape[CompressionVectorTapeIndex(params, batch, operation.tape,
                                           state)] = nodes[middle_base + state];
  }
}

__global__ void SeedRootSamples(const Scalar *nodes, const Scalar *uniforms,
                                std::uint32_t *assignments, Params params) {
  const std::size_t batch = blockIdx.x * blockDim.x + threadIdx.x;
  if (batch >= params.batch)
    return;
  assignments[AssignmentIndex(params, batch, params.root)] = SampleContiguous(
      nodes + NodeIndex(params, batch, params.root, 0), params.states,
      uniforms[AssignmentIndex(params, batch, params.root)]);
}

__global__ void ExpandSampleRakes(const btrc::Rake *operations,
                                  const Scalar *path_tape,
                                  const Scalar *leaf_tape,
                                  const Scalar *uniforms,
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
  Scalar maximum = -INFINITY;
  for (std::uint32_t child_state = 0; child_state < params.states;
       ++child_state) {
    maximum =
        fmax(maximum,
             LogProduct(path_tape[RakePathTapeIndex(
                            params, batch, operation.branch,
                            parent_state * params.states + child_state)],
                        leaf_tape[RakeLeafTapeIndex(
                            params, batch, operation.branch, child_state)]));
  }
  Scalar total = 0.0f;
  for (std::uint32_t child_state = 0; child_state < params.states;
       ++child_state) {
    total +=
        exp(LogProduct(path_tape[RakePathTapeIndex(
                           params, batch, operation.branch,
                           parent_state * params.states + child_state)],
                       leaf_tape[RakeLeafTapeIndex(
                           params, batch, operation.branch, child_state)]) -
            maximum);
  }
  const Scalar threshold =
      uniforms[AssignmentIndex(params, batch, operation.leaf)] * total;
  Scalar cumulative = 0.0f;
  std::uint32_t choice = 0;
  for (std::uint32_t child_state = 0; child_state < params.states;
       ++child_state) {
    const Scalar value =
        exp(LogProduct(path_tape[RakePathTapeIndex(
                           params, batch, operation.branch,
                           parent_state * params.states + child_state)],
                       leaf_tape[RakeLeafTapeIndex(
                           params, batch, operation.branch, child_state)]) -
            maximum);
    if (value > 0.0f)
      choice = child_state;
    cumulative += value;
    if (threshold < cumulative) {
      choice = child_state;
      break;
    }
  }
  assignments[AssignmentIndex(params, batch, operation.leaf)] = choice;
}

__global__ void
ExpandSampleCompressions(const btrc::Compression *operations,
                         const Scalar *left_tape, const Scalar *middle_tape,
                         const Scalar *right_tape, const Scalar *uniforms,
                         std::uint32_t *assignments, Params params) {
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
  Scalar maximum = -INFINITY;
  for (std::uint32_t middle_state = 0; middle_state < params.states;
       ++middle_state) {
    maximum = fmax(
        maximum, LogProduct(left_tape[CompressionMatrixTapeIndex(
                                params, batch, operation.tape,
                                parent_state * params.states + middle_state)],
                            middle_tape[CompressionVectorTapeIndex(
                                params, batch, operation.tape, middle_state)],
                            right_tape[CompressionMatrixTapeIndex(
                                params, batch, operation.tape,
                                middle_state * params.states + child_state)]));
  }
  Scalar total = 0.0f;
  for (std::uint32_t middle_state = 0; middle_state < params.states;
       ++middle_state) {
    total += exp(LogProduct(left_tape[CompressionMatrixTapeIndex(
                                params, batch, operation.tape,
                                parent_state * params.states + middle_state)],
                            middle_tape[CompressionVectorTapeIndex(
                                params, batch, operation.tape, middle_state)],
                            right_tape[CompressionMatrixTapeIndex(
                                params, batch, operation.tape,
                                middle_state * params.states + child_state)]) -
                 maximum);
  }
  const Scalar threshold =
      uniforms[AssignmentIndex(params, batch, operation.middle)] * total;
  Scalar cumulative = 0.0f;
  std::uint32_t choice = 0;
  for (std::uint32_t middle_state = 0; middle_state < params.states;
       ++middle_state) {
    const Scalar value =
        exp(LogProduct(left_tape[CompressionMatrixTapeIndex(
                           params, batch, operation.tape,
                           parent_state * params.states + middle_state)],
                       middle_tape[CompressionVectorTapeIndex(
                           params, batch, operation.tape, middle_state)],
                       right_tape[CompressionMatrixTapeIndex(
                           params, batch, operation.tape,
                           middle_state * params.states + child_state)]) -
            maximum);
    if (value > 0.0f)
      choice = middle_state;
    cumulative += value;
    if (threshold < cumulative) {
      choice = middle_state;
      break;
    }
  }
  assignments[AssignmentIndex(params, batch, operation.middle)] = choice;
}

__global__ void MaximumRakeSerial(const btrc::Rake *operations,
                                  const Scalar *nodes, const Scalar *paths,
                                  const btrc::Index *mutable_path_slots,
                                  Scalar *branches, std::uint32_t *choices,
                                  Params params) {
  const SmallStateWorkItem work = SmallStateWork();
  if (work.operation >= params.operation_count || work.batch >= params.batch)
    return;
  const btrc::Rake operation =
      operations[params.operation_offset + work.operation];
  const std::size_t node_base =
      NodeIndex(params, work.batch, operation.leaf, 0);
  const std::size_t path_base =
      PathIndex(params, mutable_path_slots, work.batch, operation.edge, 0, 0);
  for (std::size_t parent_state = 0; parent_state < params.states;
       ++parent_state) {
    Scalar best =
        paths[path_base + parent_state * params.states] + nodes[node_base];
    std::uint32_t choice = 0;
    for (std::uint32_t child_state = 1; child_state < params.states;
         ++child_state) {
      const Scalar candidate =
          paths[path_base + parent_state * params.states + child_state] +
          nodes[node_base + child_state];
      if (candidate > best) {
        best = candidate;
        choice = child_state;
      }
    }
    branches[BranchIndex(params, work.batch, operation.branch, parent_state)] =
        best;
    choices[RakeChoiceIndex(params, work.batch, operation.branch,
                            parent_state)] = choice;
  }
}

__global__ void MaximumRake(const btrc::Rake *operations, const Scalar *nodes,
                            const Scalar *paths,
                            const btrc::Index *mutable_path_slots,
                            Scalar *branches, std::uint32_t *choices,
                            Params params) {
  const std::size_t parent_state = threadIdx.x;
  const std::size_t batch = blockIdx.x % params.batch;
  const std::size_t operation_in_batch = blockIdx.x / params.batch;
  if (parent_state >= params.states || batch >= params.batch)
    return;
  const btrc::Rake operation =
      operations[params.operation_offset + operation_in_batch];
  const std::size_t node_base = NodeIndex(params, batch, operation.leaf, 0);
  const std::size_t path_base =
      PathIndex(params, mutable_path_slots, batch, operation.edge, 0, 0);
  Scalar best =
      paths[path_base + parent_state * params.states] + nodes[node_base];
  std::uint32_t choice = 0;
  for (std::uint32_t child_state = 1; child_state < params.states;
       ++child_state) {
    const Scalar candidate =
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
LogCombineBranchesSerial(const btrc::BranchCombination *operations,
                         Scalar *branches, Params params) {
  const SmallStateWorkItem work = SmallStateWork();
  if (work.operation >= params.operation_count || work.batch >= params.batch)
    return;
  const btrc::BranchCombination operation =
      operations[params.operation_offset + work.operation];
  for (std::size_t state = 0; state < params.states; ++state) {
    branches[BranchIndex(params, work.batch, operation.destination, state)] +=
        branches[BranchIndex(params, work.batch, operation.source, state)];
  }
}

__global__ void LogCombineBranches(const btrc::BranchCombination *operations,
                                   Scalar *branches, Params params) {
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
LogAbsorbBranchesSerial(const btrc::BranchAbsorption *operations, Scalar *nodes,
                        const Scalar *branches, Params params) {
  const SmallStateWorkItem work = SmallStateWork();
  if (work.operation >= params.operation_count || work.batch >= params.batch)
    return;
  const btrc::BranchAbsorption operation =
      operations[params.operation_offset + work.operation];
  for (std::size_t state = 0; state < params.states; ++state) {
    nodes[NodeIndex(params, work.batch, operation.parent, state)] +=
        branches[BranchIndex(params, work.batch, operation.branch, state)];
  }
}

__global__ void LogAbsorbBranches(const btrc::BranchAbsorption *operations,
                                  Scalar *nodes, const Scalar *branches,
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
                                       const Scalar *nodes, Scalar *paths,
                                       const btrc::Index *mutable_path_slots,
                                       std::uint32_t *choices, Params params) {
  const SmallStateWorkItem work = SmallStateWork();
  if (work.operation >= params.operation_count || work.batch >= params.batch)
    return;
  const btrc::Compression operation =
      operations[params.operation_offset + work.operation];
  constexpr std::size_t kStates = 4;
  constexpr std::size_t kMatrix = kStates * kStates;
  Scalar left[kMatrix];
  Scalar right[kMatrix];
  Scalar middle[kStates];
  const std::size_t left_base =
      PathIndex(params, mutable_path_slots, work.batch, operation.left_edge, 0,
                0);
  const std::size_t right_base =
      PathIndex(params, mutable_path_slots, work.batch, operation.right_edge, 0,
                0);
  const std::size_t middle_base =
      NodeIndex(params, work.batch, operation.middle, 0);
  for (std::size_t entry = 0; entry < kMatrix; ++entry) {
    left[entry] = paths[left_base + entry];
    right[entry] = paths[right_base + entry];
  }
  for (std::size_t state = 0; state < kStates; ++state)
    middle[state] = nodes[middle_base + state];
  for (std::size_t parent_state = 0; parent_state < kStates; ++parent_state) {
    for (std::size_t child_state = 0; child_state < kStates; ++child_state) {
      Scalar best =
          left[parent_state * kStates] + middle[0] + right[child_state];
      std::uint32_t choice = 0;
      for (std::uint32_t middle_state = 1; middle_state < kStates;
           ++middle_state) {
        const Scalar candidate = left[parent_state * kStates + middle_state] +
                                 middle[middle_state] +
                                 right[middle_state * kStates + child_state];
        if (candidate > best) {
          best = candidate;
          choice = middle_state;
        }
      }
      paths[left_base + parent_state * kStates + child_state] = best;
      choices[CompressionChoiceIndex(params, work.batch, operation.tape,
                                     parent_state, child_state)] = choice;
    }
  }
}

__global__ void MaximumCompress(const btrc::Compression *operations,
                                const Scalar *nodes, Scalar *paths,
                                const btrc::Index *mutable_path_slots,
                                std::uint32_t *choices, Params params) {
  extern __shared__ Scalar storage[];
  const std::size_t matrix =
      static_cast<std::size_t>(params.states) * params.states;
  Scalar *left = storage;
  Scalar *right = left + matrix;
  Scalar *middle = right + matrix;
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
    left[entry] =
        paths[PathIndex(params, mutable_path_slots, batch, operation.left_edge,
                        parent_state, child_state)];
    right[entry] =
        paths[PathIndex(params, mutable_path_slots, batch, operation.right_edge,
                        parent_state, child_state)];
  }
  if (entry < params.states)
    middle[entry] = nodes[NodeIndex(params, batch, operation.middle, entry)];
  __syncthreads();
  if (entry >= matrix)
    return;

  const std::size_t parent_state = entry / params.states;
  const std::size_t child_state = entry % params.states;
  Scalar best =
      left[parent_state * params.states] + middle[0] + right[child_state];
  std::uint32_t choice = 0;
  for (std::uint32_t middle_state = 1; middle_state < params.states;
       ++middle_state) {
    const Scalar candidate = left[parent_state * params.states + middle_state] +
                             middle[middle_state] +
                             right[middle_state * params.states + child_state];
    if (candidate > best) {
      best = candidate;
      choice = middle_state;
    }
  }
  paths[PathIndex(params, mutable_path_slots, batch, operation.left_edge,
                  parent_state, child_state)] = best;
  choices[CompressionChoiceIndex(params, batch, operation.tape, parent_state,
                                 child_state)] = choice;
}

__global__ void FinishMaximum(const Scalar *nodes, Scalar *log_weights,
                              std::uint32_t *assignments, Params params) {
  const std::size_t batch = blockIdx.x * blockDim.x + threadIdx.x;
  if (batch >= params.batch)
    return;
  const std::size_t root_base = NodeIndex(params, batch, params.root, 0);
  Scalar best = nodes[root_base];
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

__global__ void Rake(const btrc::Rake *operations, const Scalar *nodes,
                     const Scalar *paths, Scalar *branches,
                     const btrc::Index *mutable_path_slots,
                     const Scalar *node_scales, const Scalar *path_scales,
                     Scalar *branch_scales, Params params) {
  __shared__ Scalar normalizer;
  const std::size_t state = threadIdx.x;
  const std::size_t batch = blockIdx.x % params.batch;
  const std::size_t operation_in_batch = blockIdx.x / params.batch;
  const btrc::Rake operation =
      operations[params.operation_offset + operation_in_batch];
  if (state >= params.states || batch >= params.batch)
    return;
  const Scalar value =
      accelerator_detail::RakeValue(paths + PathIndex(params, mutable_path_slots, batch,
                                          operation.edge, 0, 0),
                        nodes + NodeIndex(params, batch, operation.leaf, 0),
                        params.states, state);
  const std::size_t branch_base =
      BranchIndex(params, batch, operation.branch, 0);
  branches[branch_base + state] = value;
  if (!params.scaled)
    return;
  __syncthreads();
  if (state == 0) {
    const Scalar maximum =
        accelerator_detail::Maximum(branches + branch_base, params.states);
    normalizer = maximum > 0.0f ? maximum : 1.0f;
    const Scalar input_scale =
        node_scales[NodeScaleIndex(params, batch, operation.leaf)] +
        path_scales[PathScaleIndex(params, mutable_path_slots, batch,
                                   operation.edge)];
    branch_scales[BranchScaleIndex(params, batch, operation.branch)] =
        accelerator_detail::UpdatedLogScale(input_scale, maximum);
  }
  __syncthreads();
  branches[branch_base + state] = value / normalizer;
}

__global__ void RakeSerial(const btrc::Rake *operations, const Scalar *nodes,
                           const Scalar *paths, Scalar *branches,
                           const btrc::Index *mutable_path_slots,
                           const Scalar *node_scales, const Scalar *path_scales,
                           Scalar *branch_scales, Params params) {
  const SmallStateWorkItem work = SmallStateWork();
  if (work.operation >= params.operation_count || work.batch >= params.batch)
    return;
  const btrc::Rake operation =
      operations[params.operation_offset + work.operation];
  const std::size_t node_base =
      NodeIndex(params, work.batch, operation.leaf, 0);
  const std::size_t path_base =
      PathIndex(params, mutable_path_slots, work.batch, operation.edge, 0, 0);
  const std::size_t branch_base =
      BranchIndex(params, work.batch, operation.branch, 0);
  Scalar values[kSmallStateLimit];
  Scalar maximum = 0.0f;
  for (std::size_t parent_state = 0; parent_state < params.states;
       ++parent_state) {
    const Scalar value = accelerator_detail::RakeValue(paths + path_base, nodes + node_base,
                                           params.states, parent_state);
    values[parent_state] = value;
    maximum = fmax(maximum, value);
  }
  const Scalar normalizer = maximum > 0.0f ? maximum : 1.0f;
  for (std::size_t state = 0; state < params.states; ++state)
    branches[branch_base + state] =
        params.scaled ? values[state] / normalizer : values[state];
  if (!params.scaled)
    return;
  const Scalar input_scale =
      node_scales[NodeScaleIndex(params, work.batch, operation.leaf)] +
      path_scales[PathScaleIndex(params, mutable_path_slots, work.batch,
                                 operation.edge)];
  branch_scales[BranchScaleIndex(params, work.batch, operation.branch)] =
      accelerator_detail::UpdatedLogScale(input_scale, maximum);
}

__global__ void CombineBranches(const btrc::BranchCombination *operations,
                                Scalar *branches, Scalar *branch_scales,
                                Params params) {
  __shared__ Scalar normalizer;
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
  const Scalar value = accelerator_detail::Product(branches[destination_base + state],
                                       branches[source_base + state]);
  branches[destination_base + state] = value;
  if (!params.scaled)
    return;
  __syncthreads();
  if (state == 0) {
    const Scalar maximum =
        accelerator_detail::Maximum(branches + destination_base, params.states);
    normalizer = maximum > 0.0f ? maximum : 1.0f;
    const std::size_t destination_scale =
        BranchScaleIndex(params, batch, operation.destination);
    const std::size_t source_scale =
        BranchScaleIndex(params, batch, operation.source);
    const Scalar input_scale =
        branch_scales[destination_scale] + branch_scales[source_scale];
    branch_scales[destination_scale] =
        accelerator_detail::UpdatedLogScale(input_scale, maximum);
  }
  __syncthreads();
  branches[destination_base + state] = value / normalizer;
}

__global__ void CombineBranchesSerial(const btrc::BranchCombination *operations,
                                      Scalar *branches, Scalar *branch_scales,
                                      Params params) {
  const SmallStateWorkItem work = SmallStateWork();
  if (work.operation >= params.operation_count || work.batch >= params.batch)
    return;
  const btrc::BranchCombination operation =
      operations[params.operation_offset + work.operation];
  const std::size_t destination_base =
      BranchIndex(params, work.batch, operation.destination, 0);
  const std::size_t source_base =
      BranchIndex(params, work.batch, operation.source, 0);
  Scalar values[kSmallStateLimit];
  Scalar maximum = 0.0f;
  for (std::size_t state = 0; state < params.states; ++state) {
    const Scalar value = accelerator_detail::Product(branches[destination_base + state],
                                         branches[source_base + state]);
    values[state] = value;
    maximum = fmax(maximum, value);
  }
  const Scalar normalizer = maximum > 0.0f ? maximum : 1.0f;
  for (std::size_t state = 0; state < params.states; ++state)
    branches[destination_base + state] =
        params.scaled ? values[state] / normalizer : values[state];
  if (!params.scaled)
    return;
  const std::size_t destination_scale =
      BranchScaleIndex(params, work.batch, operation.destination);
  const std::size_t source_scale =
      BranchScaleIndex(params, work.batch, operation.source);
  const Scalar input_scale =
      branch_scales[destination_scale] + branch_scales[source_scale];
  branch_scales[destination_scale] =
      accelerator_detail::UpdatedLogScale(input_scale, maximum);
}

__global__ void AbsorbBranches(const btrc::BranchAbsorption *operations,
                               Scalar *nodes, const Scalar *branches,
                               Scalar *node_scales, const Scalar *branch_scales,
                               Params params) {
  __shared__ Scalar normalizer;
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
  const Scalar value =
      accelerator_detail::Product(nodes[node_base + state], branches[branch_base + state]);
  nodes[node_base + state] = value;
  if (!params.scaled)
    return;
  __syncthreads();
  if (state == 0) {
    const Scalar maximum = accelerator_detail::Maximum(nodes + node_base, params.states);
    normalizer = maximum > 0.0f ? maximum : 1.0f;
    const std::size_t node_scale =
        NodeScaleIndex(params, batch, operation.parent);
    const Scalar input_scale =
        node_scales[node_scale] +
        branch_scales[BranchScaleIndex(params, batch, operation.branch)];
    node_scales[node_scale] = accelerator_detail::UpdatedLogScale(input_scale, maximum);
  }
  __syncthreads();
  nodes[node_base + state] = value / normalizer;
}

__global__ void AbsorbBranchesSerial(const btrc::BranchAbsorption *operations,
                                     Scalar *nodes, const Scalar *branches,
                                     Scalar *node_scales,
                                     const Scalar *branch_scales,
                                     Params params) {
  const SmallStateWorkItem work = SmallStateWork();
  if (work.operation >= params.operation_count || work.batch >= params.batch)
    return;
  const btrc::BranchAbsorption operation =
      operations[params.operation_offset + work.operation];
  const std::size_t node_base =
      NodeIndex(params, work.batch, operation.parent, 0);
  const std::size_t branch_base =
      BranchIndex(params, work.batch, operation.branch, 0);
  Scalar values[kSmallStateLimit];
  Scalar maximum = 0.0f;
  for (std::size_t state = 0; state < params.states; ++state) {
    const Scalar value = accelerator_detail::Product(nodes[node_base + state],
                                         branches[branch_base + state]);
    values[state] = value;
    maximum = fmax(maximum, value);
  }
  const Scalar normalizer = maximum > 0.0f ? maximum : 1.0f;
  for (std::size_t state = 0; state < params.states; ++state)
    nodes[node_base + state] =
        params.scaled ? values[state] / normalizer : values[state];
  if (!params.scaled)
    return;
  const std::size_t node_scale =
      NodeScaleIndex(params, work.batch, operation.parent);
  const Scalar input_scale =
      node_scales[node_scale] +
      branch_scales[BranchScaleIndex(params, work.batch, operation.branch)];
  node_scales[node_scale] =
      accelerator_detail::UpdatedLogScale(input_scale, maximum);
}

__global__ void Compress(const btrc::Compression *operations,
                         const Scalar *nodes, Scalar *paths,
                         const btrc::Index *mutable_path_slots,
                         const Scalar *node_scales, Scalar *path_scales,
                         Params params) {
  extern __shared__ Scalar storage[];
  __shared__ Scalar normalizer;
  const std::size_t matrix =
      static_cast<std::size_t>(params.states) * params.states;
  Scalar *left = storage;
  Scalar *right = left + matrix;
  Scalar *middle = right + matrix;
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
    left[entry] =
        paths[PathIndex(params, mutable_path_slots, batch, operation.left_edge,
                        parent_state, child_state)];
    right[entry] =
        paths[PathIndex(params, mutable_path_slots, batch, operation.right_edge,
                        parent_state, child_state)];
  }
  if (entry < params.states) {
    middle[entry] = nodes[NodeIndex(params, batch, operation.middle, entry)];
  }
  __syncthreads();
  if (entry < matrix) {
    const std::size_t middle_state = entry / params.states;
    right[entry] *= middle[middle_state];
  }
  __syncthreads();
  if (entry >= matrix)
    return;
  const std::size_t parent_state = entry / params.states;
  const std::size_t child_state = entry % params.states;
  const Scalar value = accelerator_detail::MatrixProductValue(left, right, params.states,
                                                  parent_state, child_state);
  paths[PathIndex(params, mutable_path_slots, batch, operation.left_edge,
                  parent_state, child_state)] = value;
  if (!params.scaled)
    return;
  __syncthreads();
  if (entry == 0) {
    const std::size_t output_base =
        PathIndex(params, mutable_path_slots, batch, operation.left_edge, 0, 0);
    const Scalar maximum =
        accelerator_detail::Maximum(paths + output_base, static_cast<unsigned>(matrix));
    normalizer = maximum > 0.0f ? maximum : 1.0f;
    const std::size_t left_scale =
        PathScaleIndex(params, mutable_path_slots, batch, operation.left_edge);
    const Scalar input_scale =
        path_scales[left_scale] +
        node_scales[NodeScaleIndex(params, batch, operation.middle)] +
        path_scales[PathScaleIndex(params, mutable_path_slots, batch,
                                   operation.right_edge)];
    path_scales[left_scale] = accelerator_detail::UpdatedLogScale(input_scale, maximum);
  }
  __syncthreads();
  paths[PathIndex(params, mutable_path_slots, batch, operation.left_edge,
                  parent_state, child_state)] = value / normalizer;
}

__global__ void CompressSerial4(const btrc::Compression *operations,
                                const Scalar *nodes, Scalar *paths,
                                const btrc::Index *mutable_path_slots,
                                const Scalar *node_scales, Scalar *path_scales,
                                Params params) {
  const SmallStateWorkItem work = SmallStateWork();
  if (work.operation >= params.operation_count || work.batch >= params.batch)
    return;
  const btrc::Compression operation =
      operations[params.operation_offset + work.operation];
  constexpr std::size_t kStates = 4;
  constexpr std::size_t kMatrix = kStates * kStates;
  Scalar left[kMatrix];
  Scalar right[kMatrix];
  Scalar middle[kStates];
  Scalar output[kMatrix];
  const std::size_t left_base = PathIndex(
      params, mutable_path_slots, work.batch, operation.left_edge, 0, 0);
  const std::size_t right_base = PathIndex(
      params, mutable_path_slots, work.batch, operation.right_edge, 0, 0);
  const std::size_t middle_base =
      NodeIndex(params, work.batch, operation.middle, 0);
  for (std::size_t entry = 0; entry < kMatrix; ++entry) {
    left[entry] = paths[left_base + entry];
    right[entry] = paths[right_base + entry];
  }
  for (std::size_t state = 0; state < kStates; ++state)
    middle[state] = nodes[middle_base + state];
  for (std::size_t entry = 0; entry < kMatrix; ++entry)
    right[entry] *= middle[entry / kStates];

  Scalar maximum = 0.0f;
  for (std::size_t parent_state = 0; parent_state < kStates; ++parent_state) {
    for (std::size_t child_state = 0; child_state < kStates; ++child_state) {
      const Scalar value = accelerator_detail::MatrixProductValue(
          left, right, kStates, parent_state, child_state);
      output[parent_state * kStates + child_state] = value;
      maximum = fmax(maximum, value);
    }
  }
  const Scalar normalizer = maximum > 0.0f ? maximum : 1.0f;
  for (std::size_t entry = 0; entry < kMatrix; ++entry)
    paths[left_base + entry] =
        params.scaled ? output[entry] / normalizer : output[entry];
  if (!params.scaled)
    return;
  const std::size_t left_scale =
      PathScaleIndex(params, mutable_path_slots, work.batch,
                     operation.left_edge);
  const Scalar input_scale =
      path_scales[left_scale] +
      node_scales[NodeScaleIndex(params, work.batch, operation.middle)] +
      path_scales[PathScaleIndex(params, mutable_path_slots, work.batch,
                                 operation.right_edge)];
  path_scales[left_scale] =
      accelerator_detail::UpdatedLogScale(input_scale, maximum);
}

__global__ void FinishRoot(const Scalar *nodes, const Scalar *node_scales,
                           Scalar *output, Params params) {
  const std::size_t batch = blockIdx.x;
  if (batch >= params.batch || threadIdx.x != 0)
    return;
  Scalar value = 0.0f;
  for (std::size_t state = 0; state < params.states; ++state)
    value += nodes[NodeIndex(params, batch, params.root, state)];
  output[batch] =
      params.scaled
          ? (value > 0.0f
                 ? node_scales[NodeScaleIndex(params, batch, params.root)] +
                       log(value)
                 : -INFINITY)
          : value;
}

std::uint32_t Blocks(std::size_t count, std::size_t block_size) {
  return CheckedU32((count + block_size - 1) / block_size,
                    "accelerator grid size");
}

struct SmallStateLaunch {
  dim3 grid;
  dim3 block;
};

SmallStateLaunch SmallStateLaunchDimensions(std::size_t operations,
                                            std::size_t batch,
                                            std::size_t threads) {
  constexpr std::uint32_t kMaximumGridHeight = 65535;
  std::size_t batch_threads = 1;
  while (batch_threads < std::min(batch, threads))
    batch_threads *= 2;
  const std::size_t operation_threads = threads / batch_threads;
  const std::uint32_t batch_blocks = Blocks(batch, batch_threads);
  const std::uint32_t height = std::min(batch_blocks, kMaximumGridHeight);
  const std::uint32_t depth =
      Blocks(batch_blocks, kMaximumGridHeight);
  if (depth > kMaximumGridHeight)
    throw std::length_error("small-state batch grid exceeds accelerator limits");
  return {dim3(Blocks(operations, operation_threads), height, depth),
          dim3(CheckedU32(batch_threads, "small-state batch block"),
               CheckedU32(operation_threads, "small-state operation block"))};
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
  btrc::Index *mutable_path_slots = nullptr;
  btrc::Index *mutable_path_edges = nullptr;
  Scalar *host_nodes = nullptr;
  Scalar *host_edges = nullptr;
  Scalar *host_output = nullptr;
  std::uint32_t *host_assignments = nullptr;
  Scalar *host_uniforms = nullptr;
  Scalar *host_node_marginals = nullptr;
  Scalar *host_edge_marginals = nullptr;
  std::uint8_t *host_observations = nullptr;
  Scalar *host_root_potential = nullptr;
  Scalar *host_emission_potentials = nullptr;
  Scalar *input_nodes = nullptr;
  Scalar *input_edges = nullptr;
  std::uint8_t *input_observations = nullptr;
  Scalar *input_root_potential = nullptr;
  Scalar *input_emission_potentials = nullptr;
  btrc::Index *categorical_observation_nodes = nullptr;
  btrc::Index *categorical_observation_index_by_node = nullptr;
  Scalar *nodes = nullptr;
  Scalar *paths = nullptr;
  Scalar *branches = nullptr;
  Scalar *node_scales = nullptr;
  Scalar *path_scales = nullptr;
  Scalar *branch_scales = nullptr;
  Scalar *output = nullptr;
  std::uint32_t *rake_choices = nullptr;
  std::uint32_t *compression_choices = nullptr;
  std::uint32_t *assignments = nullptr;
  Scalar *uniforms = nullptr;
  Scalar *rake_path_tape = nullptr;
  Scalar *rake_leaf_tape = nullptr;
  Scalar *compression_left_tape = nullptr;
  Scalar *compression_middle_tape = nullptr;
  Scalar *compression_right_tape = nullptr;
  Scalar *rake_message_tape = nullptr;
  Scalar *compression_output_tape = nullptr;
  Scalar *node_marginals = nullptr;
  Scalar *edge_marginals = nullptr;
  Scalar *branch_marginals = nullptr;
  std::vector<btrc::Index> observation_nodes;
  std::size_t categories = 0;
  bool categorical = false;
  bool maximum = false;
  bool sampling = false;
  bool marginals = false;
  std::size_t resident_observation_batch = 0;
  bool resident_categorical_factors = false;

  void Clear() noexcept {
    if (stream != nullptr) {
      static_cast<void>(cudaSetDevice(device));
      static_cast<void>(cudaStreamSynchronize(stream));
    }
    DeviceFree(rakes);
    DeviceFree(combinations);
    DeviceFree(absorptions);
    DeviceFree(compressions);
    DeviceFree(mutable_path_slots);
    DeviceFree(mutable_path_edges);
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
    DeviceFree(uniforms);
    DeviceFree(rake_path_tape);
    DeviceFree(rake_leaf_tape);
    DeviceFree(compression_left_tape);
    DeviceFree(compression_middle_tape);
    DeviceFree(compression_right_tape);
    DeviceFree(rake_message_tape);
    DeviceFree(compression_output_tape);
    DeviceFree(node_marginals);
    DeviceFree(edge_marginals);
    DeviceFree(branch_marginals);
    HostFree(host_nodes);
    HostFree(host_edges);
    HostFree(host_output);
    HostFree(host_assignments);
    HostFree(host_uniforms);
    HostFree(host_node_marginals);
    HostFree(host_edge_marginals);
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
    maximum = false;
    sampling = false;
    marginals = false;
    resident_observation_batch = 0;
    resident_categorical_factors = false;
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
        "GPU runtime GetDeviceProperties");
  std::ostringstream result;
#if defined(TREE_HMM_GPU_BACKEND_CUDA)
  result << properties.name << " (compute " << properties.major << '.'
         << properties.minor << ')';
#else
  result << properties.name << " (" << properties.gcnArchName << ')';
#endif
  return result.str();
}

namespace {

void ReserveCommon(Workspace::Impl &storage, const btrc::Plan &plan,
                   std::size_t states, std::size_t batch, int device) {
  if (states == 0 || batch == 0)
    throw std::invalid_argument("accelerator state and batch counts must be nonzero");
  storage.Clear();
  Check(cudaSetDevice(device), "GPU runtime SetDevice");
  cudaDeviceProp properties{};
  Check(cudaGetDeviceProperties(&properties, device),
        "GPU runtime GetDeviceProperties");
  const std::size_t matrix = CheckedProduct({states, states}, "state matrix");
  if (matrix > static_cast<std::size_t>(properties.maxThreadsPerBlock)) {
    throw std::invalid_argument(
        "state count exceeds the accelerator compression block limit");
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
      0,
  };

  Check(cudaStreamCreateWithFlags(&storage.stream, cudaStreamNonBlocking),
        "GPU runtime StreamCreateWithFlags");
  Check(cudaEventCreate(&storage.upload_start), "GPU runtime EventCreate");
  Check(cudaEventCreate(&storage.upload_stop), "GPU runtime EventCreate");
  Check(cudaEventCreate(&storage.kernel_start), "GPU runtime EventCreate");
  Check(cudaEventCreate(&storage.kernel_stop), "GPU runtime EventCreate");
  Check(cudaEventCreate(&storage.download_start), "GPU runtime EventCreate");
  Check(cudaEventCreate(&storage.download_stop), "GPU runtime EventCreate");

  DeviceAllocate(storage.rakes, plan.rakes().size());
  DeviceAllocate(storage.combinations, plan.branch_combinations().size());
  DeviceAllocate(storage.absorptions, plan.branch_absorptions().size());
  DeviceAllocate(storage.compressions, plan.compressions().size());
  Upload(storage.rakes, plan.rakes(), storage.stream);
  Upload(storage.combinations, plan.branch_combinations(), storage.stream);
  Upload(storage.absorptions, plan.branch_absorptions(), storage.stream);
  Upload(storage.compressions, plan.compressions(), storage.stream);
  const tree_hmm::internal::AcceleratorPathStorage path_storage =
      tree_hmm::internal::MakeAcceleratorPathStorage(plan);
  DeviceAllocate(storage.mutable_path_slots,
                 path_storage.mutable_slot_by_edge.size());
  DeviceAllocate(storage.mutable_path_edges,
                 path_storage.edge_by_mutable_slot.size());
  Upload(storage.mutable_path_slots, path_storage.mutable_slot_by_edge,
         storage.stream);
  Upload(storage.mutable_path_edges, path_storage.edge_by_mutable_slot,
         storage.stream);
  storage.params.mutable_paths = CheckedU32(
      path_storage.edge_by_mutable_slot.size(), "mutable path count");

  const std::size_t node_values =
      CheckedProduct({batch, plan.num_nodes(), states}, "node workspace");
  const std::size_t edge_inputs =
      CheckedProduct({plan.num_edges(), matrix}, "edge inputs");
  const std::size_t path_count = tree_hmm::internal::AcceleratorPathCount(
      plan.num_edges(), path_storage.edge_by_mutable_slot.size(), batch);
  const std::size_t path_values =
      CheckedProduct({path_count, matrix}, "path workspace");
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
  DeviceAllocate(storage.path_scales, path_count);
  DeviceAllocate(storage.branch_scales,
                 CheckedProduct({batch, plan.num_branches()}, "branch scales"));
  DeviceAllocate(storage.output, batch);
}

void ReserveAssignments(Workspace::Impl &storage) {
  const btrc::Plan &plan = *storage.plan;
  HostAllocate(
      storage.host_assignments,
      CheckedProduct({storage.batch, plan.num_nodes()}, "host assignments"));
  DeviceAllocate(
      storage.assignments,
      CheckedProduct({storage.batch, plan.num_nodes()}, "assignments"));
}

void ReserveMaximumRecovery(Workspace::Impl &storage) {
  ReserveAssignments(storage);
  const btrc::Plan &plan = *storage.plan;
  DeviceAllocate(
      storage.rake_choices,
      CheckedProduct({storage.batch, plan.num_branches(), storage.states},
                     "MAP rake choices"));
  DeviceAllocate(storage.compression_choices,
                 CheckedProduct({storage.batch, plan.num_compressions(),
                                 storage.states, storage.states},
                                "MAP compression choices"));
  storage.maximum = true;
}

void ReserveConditionalTapes(Workspace::Impl &storage) {
  const btrc::Plan &plan = *storage.plan;
  const std::size_t matrix =
      CheckedProduct({storage.states, storage.states}, "state matrix");
  const std::size_t rake_matrices = CheckedProduct(
      {storage.batch, plan.num_branches(), matrix}, "rake matrix tape");
  const std::size_t rake_vectors = CheckedProduct(
      {storage.batch, plan.num_branches(), storage.states}, "rake vector tape");
  const std::size_t compression_matrices =
      CheckedProduct({storage.batch, plan.num_compressions(), matrix},
                     "compression matrix tape");
  const std::size_t compression_vectors =
      CheckedProduct({storage.batch, plan.num_compressions(), storage.states},
                     "compression vector tape");
  DeviceAllocate(storage.rake_path_tape, rake_matrices);
  DeviceAllocate(storage.rake_leaf_tape, rake_vectors);
  DeviceAllocate(storage.compression_left_tape, compression_matrices);
  DeviceAllocate(storage.compression_middle_tape, compression_vectors);
  DeviceAllocate(storage.compression_right_tape, compression_matrices);
}

void ReserveSamplingRecovery(Workspace::Impl &storage) {
  ReserveAssignments(storage);
  ReserveConditionalTapes(storage);
  const std::size_t assignment_count = CheckedProduct(
      {storage.batch, storage.plan->num_nodes()}, "posterior uniforms");
  HostAllocate(storage.host_uniforms, assignment_count);
  DeviceAllocate(storage.uniforms, assignment_count);
  storage.sampling = true;
}

void ReserveMarginalRecovery(Workspace::Impl &storage) {
  ReserveConditionalTapes(storage);
  const btrc::Plan &plan = *storage.plan;
  const std::size_t matrix =
      CheckedProduct({storage.states, storage.states}, "state matrix");
  const std::size_t node_values = CheckedProduct(
      {storage.batch, plan.num_nodes(), storage.states}, "node marginals");
  const std::size_t edge_values = CheckedProduct(
      {storage.batch, plan.num_edges(), matrix}, "edge marginals");
  const std::size_t branch_values = CheckedProduct(
      {storage.batch, plan.num_branches(), storage.states}, "branch marginals");
  const std::size_t compression_values =
      CheckedProduct({storage.batch, plan.num_compressions(), matrix},
                     "compression output tape");
  HostAllocate(storage.host_node_marginals, node_values);
  HostAllocate(storage.host_edge_marginals, edge_values);
  DeviceAllocate(storage.node_marginals, node_values);
  DeviceAllocate(storage.edge_marginals, edge_values);
  DeviceAllocate(storage.branch_marginals, branch_values);
  DeviceAllocate(storage.rake_message_tape, branch_values);
  DeviceAllocate(storage.compression_output_tape, compression_values);
  storage.marginals = true;
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
        "accelerator categorical model must have between 1 and 256 categories");
  btrc::Index previous = 0;
  bool first = true;
  for (const btrc::Index node : observation_nodes) {
    if (node >= plan.num_nodes() || (!first && node <= previous)) {
      throw std::invalid_argument(
          "accelerator categorical observation nodes must be valid and strictly "
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

void Workspace::ReserveMaximum(const btrc::Plan &plan, std::size_t states,
                               std::size_t batch, int device) {
  Reserve(plan, states, batch, device);
  ReserveMaximumRecovery(*impl_);
}

void Workspace::ReserveCategoricalMaximum(
    const btrc::Plan &plan, std::size_t states, std::size_t batch,
    std::size_t categories, std::span<const btrc::Index> observation_nodes,
    int device) {
  ReserveCategorical(plan, states, batch, categories, observation_nodes,
                     device);
  ReserveMaximumRecovery(*impl_);
}

void Workspace::ReserveSampling(const btrc::Plan &plan, std::size_t states,
                                std::size_t batch, int device) {
  Reserve(plan, states, batch, device);
  ReserveSamplingRecovery(*impl_);
}

void Workspace::ReserveCategoricalSampling(
    const btrc::Plan &plan, std::size_t states, std::size_t batch,
    std::size_t categories, std::span<const btrc::Index> observation_nodes,
    int device) {
  ReserveCategorical(plan, states, batch, categories, observation_nodes,
                     device);
  ReserveSamplingRecovery(*impl_);
}

void Workspace::ReserveMarginals(const btrc::Plan &plan, std::size_t states,
                                 std::size_t batch, int device) {
  Reserve(plan, states, batch, device);
  ReserveMarginalRecovery(*impl_);
}

void Workspace::ReserveCategoricalMarginals(
    const btrc::Plan &plan, std::size_t states, std::size_t batch,
    std::size_t categories, std::span<const btrc::Index> observation_nodes,
    int device) {
  ReserveCategorical(plan, states, batch, categories, observation_nodes,
                     device);
  ReserveMarginalRecovery(*impl_);
}

tree_hmm::MutableBatchedModelView Workspace::Inputs() {
  return Inputs(impl_->batch);
}

tree_hmm::MutableBatchedModelView Workspace::Inputs(std::size_t batch) {
  Impl &storage = *impl_;
  if (storage.plan == nullptr)
    throw std::logic_error("accelerator Workspace::Reserve must precede Inputs");
  if (storage.categorical)
    throw std::logic_error(
        "accelerator dense Inputs cannot be used after ReserveCategorical");
  if (batch == 0 || batch > storage.batch)
    throw std::invalid_argument(
        "accelerator input batch exceeds the reserved capacity");
  const std::size_t node_values = CheckedProduct(
      {batch, storage.plan->num_nodes(), storage.states}, "accelerator input nodes");
  const std::size_t edge_values = CheckedProduct(
      {storage.plan->num_edges(), storage.states, storage.states},
      "accelerator input edges");
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
        "accelerator Workspace::ReserveCategorical must precede CategoricalInputs");
  }
  if (batch == 0 || batch > storage.batch)
    throw std::invalid_argument(
        "accelerator categorical input batch exceeds the reserved capacity");
  const std::size_t observation_values = CheckedProduct(
      {batch, storage.observation_nodes.size()}, "categorical observations");
  const std::size_t emission_values = CheckedProduct(
      {storage.categories, storage.states}, "categorical emissions");
  const std::size_t edge_values = CheckedProduct(
      {storage.plan->num_edges(), storage.states, storage.states},
      "accelerator categorical input edges");
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

std::span<Scalar> Workspace::Uniforms() { return Uniforms(impl_->batch); }

std::span<Scalar> Workspace::Uniforms(std::size_t batch) {
  Impl &storage = *impl_;
  if (!storage.sampling) {
    throw std::logic_error(
        "accelerator Workspace::ReserveSampling must precede Uniforms");
  }
  if (batch == 0 || batch > storage.batch)
    throw std::invalid_argument(
        "accelerator uniform batch exceeds the reserved capacity");
  return {storage.host_uniforms,
          CheckedProduct({batch, storage.plan->num_nodes()},
                         "accelerator posterior uniforms")};
}

namespace {

template <class StageInputs, class UploadInputs, class InitializeNodeData>
tree_hmm::PartitionView
RunPrepared(const btrc::Plan &plan, std::size_t states, std::size_t batch,
            std::span<const Scalar> edge_potentials, Workspace::Impl &storage,
            bool scaled, std::span<const Scalar> uniforms,
            StageInputs stage_inputs, UploadInputs upload_inputs,
            InitializeNodeData initialize_node_data, bool update_edges = true) {
  const auto wall_start = Clock::now();
  if (storage.plan != &plan || storage.states != states || batch == 0 ||
      batch > storage.batch) {
    throw std::invalid_argument(
        "prepared accelerator inference requires Workspace::Reserve for this plan, "
        "state count, and batch capacity");
  }
  Check(cudaSetDevice(storage.device), "GPU runtime SetDevice");
  const std::size_t matrix = CheckedProduct({states, states}, "state matrix");
  const std::size_t edge_values =
      CheckedProduct({plan.num_edges(), matrix}, "edge inputs");
  if (edge_potentials.size() != edge_values)
    throw std::invalid_argument("accelerator edge input shape is wrong");
  const bool sampling = !uniforms.empty();
  const std::size_t assignment_count =
      CheckedProduct({batch, plan.num_nodes()}, "posterior assignments");
  if (sampling && (!storage.sampling || uniforms.size() != assignment_count)) {
    throw std::invalid_argument(
        "prepared accelerator posterior sampling requires ReserveSampling and one "
        "uniform variate per batch item and node");
  }
  for (const Scalar uniform : uniforms) {
    if (!std::isfinite(uniform) || uniform < 0.0f || uniform >= 1.0f) {
      throw std::invalid_argument(
          "posterior-sampling variates must lie in [0, 1)");
    }
  }
  stage_inputs();
  if (update_edges && edge_potentials.data() != storage.host_edges) {
    std::memcpy(storage.host_edges, edge_potentials.data(),
                edge_potentials.size_bytes());
  }
  if (sampling && uniforms.data() != storage.host_uniforms) {
    std::memcpy(storage.host_uniforms, uniforms.data(), uniforms.size_bytes());
  }

  Check(cudaEventRecord(storage.upload_start, storage.stream),
        "GPU runtime EventRecord upload start");
  upload_inputs();
  if (update_edges) {
    Check(cudaMemcpyAsync(storage.input_edges, storage.host_edges,
                          edge_potentials.size_bytes(), cudaMemcpyHostToDevice,
                          storage.stream),
          "GPU runtime MemcpyAsync edge upload");
  }
  if (sampling) {
    Check(cudaMemcpyAsync(storage.uniforms, storage.host_uniforms,
                          uniforms.size_bytes(), cudaMemcpyHostToDevice,
                          storage.stream),
          "GPU runtime MemcpyAsync posterior-uniform upload");
  }
  Check(cudaEventRecord(storage.upload_stop, storage.stream),
        "GPU runtime EventRecord upload stop");

  constexpr std::size_t kThreads = 256;
  Check(cudaEventRecord(storage.kernel_start, storage.stream),
        "GPU runtime EventRecord kernel start");
  Params base_params = storage.params;
  base_params.batch = CheckedU32(batch, "batch count");
  base_params.scaled = scaled ? 1 : 0;
  const std::size_t path_matrices = tree_hmm::internal::AcceleratorPathCount(
      base_params.edges, base_params.mutable_paths, batch);
  if (scaled) {
    Check(cudaMemsetAsync(storage.node_scales, 0,
                          batch * plan.num_nodes() * sizeof(Scalar),
                          storage.stream),
          "GPU runtime MemsetAsync node scales");
    Check(cudaMemsetAsync(storage.path_scales, 0,
                          path_matrices * sizeof(Scalar), storage.stream),
          "GPU runtime MemsetAsync path scales");
    Check(cudaMemsetAsync(storage.branch_scales, 0,
                          batch * plan.num_branches() * sizeof(Scalar),
                          storage.stream),
          "GPU runtime MemsetAsync branch scales");
  }
  initialize_node_data(base_params);
  if (path_matrices != 0) {
    InitializePaths<<<Blocks(path_matrices, kThreads), kThreads, 0,
                      storage.stream>>>(storage.input_edges,
                                        storage.mutable_path_edges,
                                        storage.paths,
                                        base_params);
  }

  for (const btrc::PrimitiveBatch &primitive_batch : plan.primitive_batches()) {
    Params params = base_params;
    params.operation_offset = primitive_batch.offset;
    params.operation_count = primitive_batch.count;
    const std::size_t operations =
        CheckedProduct({batch, primitive_batch.count}, "primitive accelerator grid");
    const std::uint32_t operation_blocks =
        CheckedU32(operations, "primitive accelerator grid");
    const std::uint32_t linear_blocks = Blocks(operations, kThreads);
    const SmallStateLaunch small_state =
        SmallStateLaunchDimensions(primitive_batch.count, batch, kThreads);
    if (sampling && primitive_batch.primitive == btrc::Primitive::kRake) {
      SaveRakeTapes<<<linear_blocks, kThreads, 0, storage.stream>>>(
          storage.rakes, storage.nodes, storage.paths,
          storage.mutable_path_slots, storage.rake_path_tape,
          storage.rake_leaf_tape, params);
    }
    if (sampling &&
        primitive_batch.primitive == btrc::Primitive::kCompression) {
      SaveCompressionTapes<<<linear_blocks, kThreads, 0, storage.stream>>>(
          storage.compressions, storage.nodes, storage.paths,
          storage.mutable_path_slots, storage.compression_left_tape,
          storage.compression_middle_tape, storage.compression_right_tape,
          params);
    }
    switch (primitive_batch.primitive) {
    case btrc::Primitive::kRake:
      if (states <= kSmallStateLimit) {
        RakeSerial<<<small_state.grid, small_state.block, 0, storage.stream>>>(
            storage.rakes, storage.nodes, storage.paths, storage.branches,
            storage.mutable_path_slots, storage.node_scales,
            storage.path_scales, storage.branch_scales, params);
      } else {
        Rake<<<operation_blocks, states, 0, storage.stream>>>(
            storage.rakes, storage.nodes, storage.paths, storage.branches,
            storage.mutable_path_slots, storage.node_scales,
            storage.path_scales, storage.branch_scales, params);
      }
      break;
    case btrc::Primitive::kBranchCombination:
      if (states <= kSmallStateLimit) {
        CombineBranchesSerial<<<small_state.grid, small_state.block, 0,
                                storage.stream>>>(
            storage.combinations, storage.branches, storage.branch_scales,
            params);
      } else {
        CombineBranches<<<operation_blocks, states, 0, storage.stream>>>(
            storage.combinations, storage.branches, storage.branch_scales,
            params);
      }
      break;
    case btrc::Primitive::kBranchAbsorption:
      if (states <= kSmallStateLimit) {
        AbsorbBranchesSerial<<<small_state.grid, small_state.block, 0,
                               storage.stream>>>(
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
        CompressSerial4<<<small_state.grid, small_state.block, 0,
                           storage.stream>>>(
            storage.compressions, storage.nodes, storage.paths,
            storage.mutable_path_slots, storage.node_scales,
            storage.path_scales, params);
      } else {
        Compress<<<operation_blocks, matrix,
                   (2 * matrix + states) * sizeof(Scalar), storage.stream>>>(
            storage.compressions, storage.nodes, storage.paths,
            storage.mutable_path_slots, storage.node_scales,
            storage.path_scales, params);
      }
      break;
    }
  }
  FinishRoot<<<CheckedU32(batch, "root accelerator grid"), 1, 0, storage.stream>>>(
      storage.nodes, storage.node_scales, storage.output, base_params);
  if (sampling) {
    SeedRootSamples<<<Blocks(batch, kThreads), kThreads, 0, storage.stream>>>(
        storage.nodes, storage.uniforms, storage.assignments, base_params);
    for (std::size_t index = plan.primitive_batches().size(); index-- > 0;) {
      const btrc::PrimitiveBatch &primitive_batch =
          plan.primitive_batches()[index];
      Params params = base_params;
      params.operation_offset = primitive_batch.offset;
      params.operation_count = primitive_batch.count;
      const std::size_t operations = CheckedProduct(
          {batch, primitive_batch.count}, "posterior expansion grid");
      switch (primitive_batch.primitive) {
      case btrc::Primitive::kRake:
        ExpandSampleRakes<<<Blocks(operations, kThreads), kThreads, 0,
                            storage.stream>>>(
            storage.rakes, storage.rake_path_tape, storage.rake_leaf_tape,
            storage.uniforms, storage.assignments, params);
        break;
      case btrc::Primitive::kCompression:
        ExpandSampleCompressions<<<Blocks(operations, kThreads), kThreads, 0,
                                   storage.stream>>>(
            storage.compressions, storage.compression_left_tape,
            storage.compression_middle_tape, storage.compression_right_tape,
            storage.uniforms, storage.assignments, params);
        break;
      case btrc::Primitive::kBranchCombination:
      case btrc::Primitive::kBranchAbsorption:
        break;
      }
    }
  }
  Check(cudaGetLastError(), "tree-HMM accelerator kernel launch");
  Check(cudaEventRecord(storage.kernel_stop, storage.stream),
        "GPU runtime EventRecord kernel stop");

  Check(cudaEventRecord(storage.download_start, storage.stream),
        "GPU runtime EventRecord download start");
  Check(cudaMemcpyAsync(storage.host_output, storage.output,
                        batch * sizeof(Scalar), cudaMemcpyDeviceToHost,
                        storage.stream),
        "GPU runtime MemcpyAsync output download");
  if (sampling) {
    Check(cudaMemcpyAsync(storage.host_assignments, storage.assignments,
                          assignment_count * sizeof(std::uint32_t),
                          cudaMemcpyDeviceToHost, storage.stream),
          "GPU runtime MemcpyAsync posterior-sample download");
  }
  Check(cudaEventRecord(storage.download_stop, storage.stream),
        "GPU runtime EventRecord download stop");
  Check(cudaEventSynchronize(storage.download_stop), "tree-HMM accelerator execution");
  if (sampling) {
    for (std::size_t batch_index = 0; batch_index < batch; ++batch_index) {
      if (!std::isfinite(storage.host_output[batch_index])) {
        throw std::domain_error(
            "the tree HMM has a nonpositive partition function");
      }
    }
  }

  float upload_ms = 0.0f;
  float kernel_ms = 0.0f;
  float download_ms = 0.0f;
  Check(cudaEventElapsedTime(&upload_ms, storage.upload_start,
                             storage.upload_stop),
        "GPU runtime EventElapsedTime upload");
  Check(cudaEventElapsedTime(&kernel_ms, storage.kernel_start,
                             storage.kernel_stop),
        "GPU runtime EventElapsedTime kernels");
  Check(cudaEventElapsedTime(&download_ms, storage.download_start,
                             storage.download_stop),
        "GPU runtime EventElapsedTime download");
  const double wall_ms =
      std::chrono::duration<double, std::milli>(Clock::now() - wall_start)
          .count();
  return {{storage.host_output, batch},
          {upload_ms, kernel_ms, download_ms, wall_ms}};
}

template <class StageInputs, class UploadInputs, class InitializeNodeData>
tree_hmm::BatchedMaximumAssignmentView
RunMaximumPrepared(const btrc::Plan &plan, std::size_t states,
                   std::size_t batch, std::span<const Scalar> edge_potentials,
                   Workspace::Impl &storage, StageInputs stage_inputs,
                   UploadInputs upload_inputs,
                   InitializeNodeData initialize_node_data,
                   bool update_edges = true) {
  const auto wall_start = Clock::now();
  if (!storage.maximum || storage.plan != &plan || storage.states != states ||
      batch == 0 || batch > storage.batch) {
    throw std::invalid_argument(
        "prepared accelerator MAP inference requires ReserveMaximum for "
        "this plan, state count, and batch capacity");
  }
  Check(cudaSetDevice(storage.device), "GPU runtime SetDevice");
  const std::size_t matrix = CheckedProduct({states, states}, "state matrix");
  const std::size_t edge_values =
      CheckedProduct({plan.num_edges(), matrix}, "edge inputs");
  if (edge_potentials.size() != edge_values)
    throw std::invalid_argument("accelerator edge input shape is wrong");
  stage_inputs();
  if (update_edges && edge_potentials.data() != storage.host_edges) {
    std::memcpy(storage.host_edges, edge_potentials.data(),
                edge_potentials.size_bytes());
  }

  Check(cudaEventRecord(storage.upload_start, storage.stream),
        "GPU runtime EventRecord upload start");
  upload_inputs();
  if (update_edges) {
    Check(cudaMemcpyAsync(storage.input_edges, storage.host_edges,
                          edge_potentials.size_bytes(), cudaMemcpyHostToDevice,
                          storage.stream),
          "GPU runtime MemcpyAsync edge upload");
  }
  Check(cudaEventRecord(storage.upload_stop, storage.stream),
        "GPU runtime EventRecord upload stop");

  constexpr std::size_t kThreads = 256;
  Check(cudaEventRecord(storage.kernel_start, storage.stream),
        "GPU runtime EventRecord kernel start");
  Params base_params = storage.params;
  base_params.batch = CheckedU32(batch, "batch count");
  base_params.scaled = 0;
  const std::size_t path_matrices = tree_hmm::internal::AcceleratorPathCount(
      base_params.edges, base_params.mutable_paths, batch);
  initialize_node_data(base_params);
  if (path_matrices != 0) {
    InitializePaths<<<Blocks(path_matrices, kThreads), kThreads, 0,
                      storage.stream>>>(storage.input_edges,
                                        storage.mutable_path_edges,
                                        storage.paths,
                                        base_params);
  }
  const std::size_t node_values =
      CheckedProduct({batch, plan.num_nodes(), states}, "MAP node values");
  TakeLogs<<<Blocks(node_values, kThreads), kThreads, 0, storage.stream>>>(
      storage.nodes, node_values);
  const std::size_t path_values =
      CheckedProduct({path_matrices, matrix}, "MAP path values");
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
    const SmallStateLaunch small_state =
        SmallStateLaunchDimensions(primitive_batch.count, batch, kThreads);
    switch (primitive_batch.primitive) {
    case btrc::Primitive::kRake:
      if (states <= kSmallStateLimit) {
        MaximumRakeSerial<<<small_state.grid, small_state.block, 0,
                            storage.stream>>>(
            storage.rakes, storage.nodes, storage.paths,
            storage.mutable_path_slots, storage.branches, storage.rake_choices,
            params);
      } else {
        MaximumRake<<<operation_blocks, states, 0, storage.stream>>>(
            storage.rakes, storage.nodes, storage.paths,
            storage.mutable_path_slots, storage.branches, storage.rake_choices,
            params);
      }
      break;
    case btrc::Primitive::kBranchCombination:
      if (states <= kSmallStateLimit) {
        LogCombineBranchesSerial<<<small_state.grid, small_state.block, 0,
                                   storage.stream>>>(storage.combinations,
                                                     storage.branches, params);
      } else {
        LogCombineBranches<<<operation_blocks, states, 0, storage.stream>>>(
            storage.combinations, storage.branches, params);
      }
      break;
    case btrc::Primitive::kBranchAbsorption:
      if (states <= kSmallStateLimit) {
        LogAbsorbBranchesSerial<<<small_state.grid, small_state.block, 0,
                                  storage.stream>>>(
            storage.absorptions, storage.nodes, storage.branches, params);
      } else {
        LogAbsorbBranches<<<operation_blocks, states, 0, storage.stream>>>(
            storage.absorptions, storage.nodes, storage.branches, params);
      }
      break;
    case btrc::Primitive::kCompression:
      if (states == 4) {
        MaximumCompressSerial4<<<small_state.grid, small_state.block, 0,
                                  storage.stream>>>(
            storage.compressions, storage.nodes, storage.paths,
            storage.mutable_path_slots, storage.compression_choices, params);
      } else {
        MaximumCompress<<<operation_blocks, matrix,
                          (2 * matrix + states) * sizeof(Scalar),
                          storage.stream>>>(
            storage.compressions, storage.nodes, storage.paths,
            storage.mutable_path_slots, storage.compression_choices, params);
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
  Check(cudaGetLastError(), "tree-HMM accelerator MAP kernel launch");
  Check(cudaEventRecord(storage.kernel_stop, storage.stream),
        "GPU runtime EventRecord kernel stop");

  Check(cudaEventRecord(storage.download_start, storage.stream),
        "GPU runtime EventRecord download start");
  Check(cudaMemcpyAsync(storage.host_output, storage.output,
                        batch * sizeof(Scalar), cudaMemcpyDeviceToHost,
                        storage.stream),
        "GPU runtime MemcpyAsync MAP-weight download");
  const std::size_t assignment_count =
      CheckedProduct({batch, plan.num_nodes()}, "MAP assignment download");
  Check(cudaMemcpyAsync(storage.host_assignments, storage.assignments,
                        assignment_count * sizeof(std::uint32_t),
                        cudaMemcpyDeviceToHost, storage.stream),
        "GPU runtime MemcpyAsync MAP-assignment download");
  Check(cudaEventRecord(storage.download_stop, storage.stream),
        "GPU runtime EventRecord download stop");
  Check(cudaEventSynchronize(storage.download_stop),
        "tree-HMM accelerator MAP execution");

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
        "GPU runtime EventElapsedTime MAP upload");
  Check(cudaEventElapsedTime(&kernel_ms, storage.kernel_start,
                             storage.kernel_stop),
        "GPU runtime EventElapsedTime MAP kernels");
  Check(cudaEventElapsedTime(&download_ms, storage.download_start,
                             storage.download_stop),
        "GPU runtime EventElapsedTime MAP download");
  const double wall_ms =
      std::chrono::duration<double, std::milli>(Clock::now() - wall_start)
          .count();
  return {{storage.host_output, batch},
          {storage.host_assignments, assignment_count},
          {upload_ms, kernel_ms, download_ms, wall_ms}};
}

template <class StageInputs, class UploadInputs, class InitializeNodeData>
tree_hmm::BatchedMarginalView
RunMarginalsPrepared(const btrc::Plan &plan, std::size_t states,
                     std::size_t batch, std::span<const Scalar> edge_potentials,
                     Workspace::Impl &storage, StageInputs stage_inputs,
                     UploadInputs upload_inputs,
                     InitializeNodeData initialize_node_data,
                     bool update_edges = true) {
  const auto wall_start = Clock::now();
  if (!storage.marginals || storage.plan != &plan || storage.states != states ||
      batch == 0 || batch > storage.batch) {
    throw std::invalid_argument(
        "prepared accelerator marginal inference requires ReserveMarginals for "
        "this plan, state count, and batch capacity");
  }
  Check(cudaSetDevice(storage.device), "GPU runtime SetDevice");
  const std::size_t matrix = CheckedProduct({states, states}, "state matrix");
  const std::size_t edge_values =
      CheckedProduct({plan.num_edges(), matrix}, "edge inputs");
  if (edge_potentials.size() != edge_values)
    throw std::invalid_argument("accelerator edge input shape is wrong");
  stage_inputs();
  if (update_edges && edge_potentials.data() != storage.host_edges) {
    std::memcpy(storage.host_edges, edge_potentials.data(),
                edge_potentials.size_bytes());
  }

  Check(cudaEventRecord(storage.upload_start, storage.stream),
        "GPU runtime EventRecord upload start");
  upload_inputs();
  if (update_edges) {
    Check(cudaMemcpyAsync(storage.input_edges, storage.host_edges,
                          edge_potentials.size_bytes(), cudaMemcpyHostToDevice,
                          storage.stream),
          "GPU runtime MemcpyAsync edge upload");
  }
  Check(cudaEventRecord(storage.upload_stop, storage.stream),
        "GPU runtime EventRecord upload stop");

  constexpr std::size_t kThreads = 256;
  Check(cudaEventRecord(storage.kernel_start, storage.stream),
        "GPU runtime EventRecord kernel start");
  Params base_params = storage.params;
  base_params.batch = CheckedU32(batch, "batch count");
  base_params.scaled = 0;
  const std::size_t path_matrices = tree_hmm::internal::AcceleratorPathCount(
      base_params.edges, base_params.mutable_paths, batch);
  initialize_node_data(base_params);
  if (path_matrices != 0) {
    InitializePaths<<<Blocks(path_matrices, kThreads), kThreads, 0,
                      storage.stream>>>(storage.input_edges,
                                        storage.mutable_path_edges,
                                        storage.paths,
                                        base_params);
  }
  const std::size_t node_values =
      CheckedProduct({batch, plan.num_nodes(), states}, "marginal node values");
  TakeLogs<<<Blocks(node_values, kThreads), kThreads, 0, storage.stream>>>(
      storage.nodes, node_values);
  const std::size_t path_values =
      CheckedProduct({path_matrices, matrix}, "marginal path values");
  if (path_values != 0) {
    TakeLogs<<<Blocks(path_values, kThreads), kThreads, 0, storage.stream>>>(
        storage.paths, path_values);
  }
  const std::size_t marginal_edge_values =
      CheckedProduct({batch, plan.num_edges(), matrix}, "edge marginals");
  const std::size_t branch_values =
      CheckedProduct({batch, plan.num_branches(), states}, "branch marginals");
  Check(cudaMemsetAsync(storage.node_marginals, 0, node_values * sizeof(Scalar),
                        storage.stream),
        "GPU runtime MemsetAsync node marginals");
  Check(cudaMemsetAsync(storage.edge_marginals, 0,
                        marginal_edge_values * sizeof(Scalar), storage.stream),
        "GPU runtime MemsetAsync edge marginals");
  Check(cudaMemsetAsync(storage.branch_marginals, 0,
                        branch_values * sizeof(Scalar), storage.stream),
        "GPU runtime MemsetAsync branch marginals");

  for (const btrc::PrimitiveBatch &primitive_batch : plan.primitive_batches()) {
    Params params = base_params;
    params.operation_offset = primitive_batch.offset;
    params.operation_count = primitive_batch.count;
    const std::size_t operations = CheckedProduct(
        {batch, primitive_batch.count}, "marginal primitive grid");
    const std::uint32_t operation_blocks =
        CheckedU32(operations, "marginal primitive grid");
    switch (primitive_batch.primitive) {
    case btrc::Primitive::kRake:
      LogRake<<<operation_blocks, states, 0, storage.stream>>>(
          storage.rakes, storage.nodes, storage.paths,
          storage.mutable_path_slots, storage.branches, storage.rake_path_tape,
          storage.rake_leaf_tape, storage.rake_message_tape, params);
      break;
    case btrc::Primitive::kBranchCombination:
      LogCombineBranches<<<operation_blocks, states, 0, storage.stream>>>(
          storage.combinations, storage.branches, params);
      break;
    case btrc::Primitive::kBranchAbsorption:
      LogAbsorbBranches<<<operation_blocks, states, 0, storage.stream>>>(
          storage.absorptions, storage.nodes, storage.branches, params);
      break;
    case btrc::Primitive::kCompression:
      LogCompress<<<operation_blocks, matrix,
                    (2 * matrix + states) * sizeof(Scalar), storage.stream>>>(
          storage.compressions, storage.nodes, storage.paths,
          storage.mutable_path_slots, storage.compression_left_tape,
          storage.compression_middle_tape, storage.compression_right_tape,
          storage.compression_output_tape, params);
      break;
    }
  }

  FinishLogRootAndSeedMarginals<<<Blocks(batch, kThreads), kThreads, 0,
                                  storage.stream>>>(
      storage.nodes, storage.output, storage.node_marginals, base_params);
  for (std::size_t index = plan.primitive_batches().size(); index-- > 0;) {
    const btrc::PrimitiveBatch &primitive_batch =
        plan.primitive_batches()[index];
    Params params = base_params;
    params.operation_offset = primitive_batch.offset;
    params.operation_count = primitive_batch.count;
    const std::size_t operations =
        CheckedProduct({batch, primitive_batch.count}, "marginal reverse grid");
    const std::uint32_t operation_blocks =
        CheckedU32(operations, "marginal reverse grid");
    switch (primitive_batch.primitive) {
    case btrc::Primitive::kRake:
      ReverseLogRakes<<<operation_blocks, matrix, 0, storage.stream>>>(
          storage.rakes, storage.rake_path_tape, storage.rake_leaf_tape,
          storage.rake_message_tape, storage.branch_marginals,
          storage.node_marginals, storage.edge_marginals, params);
      break;
    case btrc::Primitive::kBranchCombination:
      ReverseLogCombinations<<<operation_blocks, states, 0, storage.stream>>>(
          storage.combinations, storage.branch_marginals, params);
      break;
    case btrc::Primitive::kBranchAbsorption:
      ReverseLogAbsorptions<<<operation_blocks, states, 0, storage.stream>>>(
          storage.absorptions, storage.node_marginals, storage.branch_marginals,
          params);
      break;
    case btrc::Primitive::kCompression:
      ReverseLogCompressions<<<operation_blocks, matrix,
                               matrix * sizeof(Scalar), storage.stream>>>(
          storage.compressions, storage.compression_left_tape,
          storage.compression_middle_tape, storage.compression_right_tape,
          storage.compression_output_tape, storage.node_marginals,
          storage.edge_marginals, params);
      break;
    }
  }
  Check(cudaGetLastError(), "tree-HMM accelerator marginal kernel launch");
  Check(cudaEventRecord(storage.kernel_stop, storage.stream),
        "GPU runtime EventRecord kernel stop");

  Check(cudaEventRecord(storage.download_start, storage.stream),
        "GPU runtime EventRecord download start");
  Check(cudaMemcpyAsync(storage.host_output, storage.output,
                        batch * sizeof(Scalar), cudaMemcpyDeviceToHost,
                        storage.stream),
        "GPU runtime MemcpyAsync marginal log-partition download");
  Check(cudaMemcpyAsync(storage.host_node_marginals, storage.node_marginals,
                        node_values * sizeof(Scalar), cudaMemcpyDeviceToHost,
                        storage.stream),
        "GPU runtime MemcpyAsync node-marginal download");
  Check(cudaMemcpyAsync(storage.host_edge_marginals, storage.edge_marginals,
                        marginal_edge_values * sizeof(Scalar),
                        cudaMemcpyDeviceToHost, storage.stream),
        "GPU runtime MemcpyAsync edge-marginal download");
  Check(cudaEventRecord(storage.download_stop, storage.stream),
        "GPU runtime EventRecord download stop");
  Check(cudaEventSynchronize(storage.download_stop),
        "tree-HMM accelerator marginal execution");
  for (std::size_t batch_index = 0; batch_index < batch; ++batch_index) {
    if (!std::isfinite(storage.host_output[batch_index])) {
      throw std::domain_error(
          "the tree HMM has a nonpositive partition function");
    }
  }

  float upload_ms = 0.0f;
  float kernel_ms = 0.0f;
  float download_ms = 0.0f;
  Check(cudaEventElapsedTime(&upload_ms, storage.upload_start,
                             storage.upload_stop),
        "GPU runtime EventElapsedTime marginal upload");
  Check(cudaEventElapsedTime(&kernel_ms, storage.kernel_start,
                             storage.kernel_stop),
        "GPU runtime EventElapsedTime marginal kernels");
  Check(cudaEventElapsedTime(&download_ms, storage.download_start,
                             storage.download_stop),
        "GPU runtime EventElapsedTime marginal download");
  const double wall_ms =
      std::chrono::duration<double, std::milli>(Clock::now() - wall_start)
          .count();
  return {{storage.host_output, batch},
          {storage.host_node_marginals, node_values},
          {storage.host_edge_marginals, marginal_edge_values},
          {upload_ms, kernel_ms, download_ms, wall_ms}};
}

template <class Execute>
auto WithDenseInputs(tree_hmm::BatchedModelView model, Workspace::Impl &storage,
                     Execute execute) {
  if (storage.categorical)
    throw std::invalid_argument(
        "dense accelerator inference cannot use a categorical workspace");
  const std::size_t node_values = CheckedProduct(
      {model.batch, model.plan.num_nodes(), model.states}, "node inputs");
  if (model.node_potentials.size() != node_values)
    throw std::invalid_argument("accelerator node input shape is wrong");
  return execute(
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
              "GPU runtime MemcpyAsync node upload");
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

template <class Execute>
auto WithCategoricalInputs(tree_hmm::BatchedCategoricalModelView model,
                           Workspace::Impl &storage,
                           tree_hmm::CategoricalInputUpdate update,
                           Execute execute) {
  if (!storage.categorical || model.categories != storage.categories ||
      model.observation_nodes.size() != storage.observation_nodes.size() ||
      !std::equal(model.observation_nodes.begin(),
                  model.observation_nodes.end(),
                  storage.observation_nodes.begin())) {
    throw std::invalid_argument(
        "categorical accelerator model does not match the reserved workspace");
  }
  const std::size_t observation_values =
      CheckedProduct({model.batch, model.observation_nodes.size()},
                     "categorical observations");
  const std::size_t emission_values =
      CheckedProduct({model.categories, model.states}, "categorical emissions");
  if (model.observations.size() != observation_values ||
      model.root_potential.size() != model.states ||
      model.emission_potentials.size() != emission_values) {
    throw std::invalid_argument("accelerator categorical input shapes are wrong");
  }
  const bool update_observations =
      update == tree_hmm::CategoricalInputUpdate::kAll;
  const bool update_factors =
      update != tree_hmm::CategoricalInputUpdate::kNone;
  if (!update_observations &&
      storage.resident_observation_batch != model.batch) {
    throw std::logic_error(
        "categorical observations are not resident for this batch; use kAll");
  }
  if (!update_factors && !storage.resident_categorical_factors) {
    throw std::logic_error(
        "categorical factors are not resident; use kAll or kFactors");
  }
  auto result = execute(
      [&] {
        if (update_observations &&
            model.observations.data() != storage.host_observations) {
          std::memcpy(storage.host_observations, model.observations.data(),
                      model.observations.size_bytes());
        }
        if (update_factors &&
            model.root_potential.data() != storage.host_root_potential) {
          std::memcpy(storage.host_root_potential, model.root_potential.data(),
                      model.root_potential.size_bytes());
        }
        if (update_factors && model.emission_potentials.data() !=
            storage.host_emission_potentials) {
          std::memcpy(storage.host_emission_potentials,
                      model.emission_potentials.data(),
                      model.emission_potentials.size_bytes());
        }
      },
      [&] {
        if (update_observations) {
          Check(cudaMemcpyAsync(storage.input_observations,
                                storage.host_observations,
                                model.observations.size_bytes(),
                                cudaMemcpyHostToDevice, storage.stream),
                "GPU runtime MemcpyAsync categorical observation upload");
        }
        if (update_factors) {
          Check(cudaMemcpyAsync(storage.input_root_potential,
                                storage.host_root_potential,
                                model.root_potential.size_bytes(),
                                cudaMemcpyHostToDevice, storage.stream),
                "GPU runtime MemcpyAsync root-potential upload");
          Check(cudaMemcpyAsync(storage.input_emission_potentials,
                                storage.host_emission_potentials,
                                model.emission_potentials.size_bytes(),
                                cudaMemcpyHostToDevice, storage.stream),
                "GPU runtime MemcpyAsync emission-potential upload");
        }
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
      }, update_factors);
  if (update_observations)
    storage.resident_observation_batch = model.batch;
  if (update_factors)
    storage.resident_categorical_factors = true;
  return result;
}

tree_hmm::PartitionView Run(tree_hmm::BatchedModelView model,
                            Workspace::Impl &storage, bool scaled,
                            std::span<const Scalar> uniforms = {}) {
  return WithDenseInputs(
      model, storage, [&](auto stage, auto upload, auto initialize) {
        return RunPrepared(model.plan, model.states, model.batch,
                           model.edge_potentials, storage, scaled, uniforms,
                           stage, upload, initialize);
      });
}

tree_hmm::PartitionView Run(tree_hmm::BatchedCategoricalModelView model,
                            Workspace::Impl &storage, bool scaled,
                            std::span<const Scalar> uniforms,
                            tree_hmm::CategoricalInputUpdate update) {
  return WithCategoricalInputs(
      model, storage, update, [&](auto stage, auto upload, auto initialize,
                                 bool update_edges) {
        return RunPrepared(model.plan, model.states, model.batch,
                           model.edge_potentials, storage, scaled, uniforms,
                           stage, upload, initialize, update_edges);
      });
}

tree_hmm::BatchedMaximumAssignmentView
RunMaximum(tree_hmm::BatchedModelView model, Workspace::Impl &storage) {
  return WithDenseInputs(
      model, storage, [&](auto stage, auto upload, auto initialize) {
        return RunMaximumPrepared(model.plan, model.states, model.batch,
                                  model.edge_potentials, storage, stage, upload,
                                  initialize);
      });
}

tree_hmm::BatchedMaximumAssignmentView
RunMaximum(tree_hmm::BatchedCategoricalModelView model,
           Workspace::Impl &storage,
           tree_hmm::CategoricalInputUpdate update) {
  return WithCategoricalInputs(
      model, storage, update, [&](auto stage, auto upload, auto initialize,
                                 bool update_edges) {
        return RunMaximumPrepared(model.plan, model.states, model.batch,
                                  model.edge_potentials, storage, stage, upload,
                                  initialize, update_edges);
      });
}

tree_hmm::BatchedMarginalView RunMarginals(tree_hmm::BatchedModelView model,
                                           Workspace::Impl &storage) {
  return WithDenseInputs(
      model, storage, [&](auto stage, auto upload, auto initialize) {
        return RunMarginalsPrepared(model.plan, model.states, model.batch,
                                    model.edge_potentials, storage, stage,
                                    upload, initialize);
      });
}

tree_hmm::BatchedMarginalView
RunMarginals(tree_hmm::BatchedCategoricalModelView model,
             Workspace::Impl &storage,
             tree_hmm::CategoricalInputUpdate update) {
  return WithCategoricalInputs(
      model, storage, update, [&](auto stage, auto upload, auto initialize,
                                 bool update_edges) {
        return RunMarginalsPrepared(model.plan, model.states, model.batch,
                                    model.edge_potentials, storage, stage,
                                    upload, initialize, update_edges);
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
                          Workspace &workspace,
                          tree_hmm::CategoricalInputUpdate update) {
  return Run(model, *workspace.impl_, false, {}, update);
}

tree_hmm::PartitionView
LogPartitionFunctionPrepared(tree_hmm::BatchedCategoricalModelView model,
                             Workspace &workspace,
                             tree_hmm::CategoricalInputUpdate update) {
  return Run(model, *workspace.impl_, true, {}, update);
}

tree_hmm::BatchedMaximumAssignmentView
MaximumAPosterioriPrepared(tree_hmm::BatchedModelView model,
                           Workspace &workspace) {
  return RunMaximum(model, *workspace.impl_);
}

tree_hmm::BatchedMaximumAssignmentView
MaximumAPosterioriPrepared(tree_hmm::BatchedCategoricalModelView model,
                           Workspace &workspace,
                           tree_hmm::CategoricalInputUpdate update) {
  return RunMaximum(model, *workspace.impl_, update);
}

tree_hmm::BatchedPosteriorSampleView
PosteriorSamplePrepared(tree_hmm::BatchedModelView model,
                        std::span<const Scalar> uniforms,
                        Workspace &workspace) {
  const tree_hmm::PartitionView result =
      Run(model, *workspace.impl_, true, uniforms);
  return {
      {workspace.impl_->host_assignments, model.batch * model.plan.num_nodes()},
      result.timings};
}

tree_hmm::BatchedPosteriorSampleView
PosteriorSamplePrepared(tree_hmm::BatchedCategoricalModelView model,
                        std::span<const Scalar> uniforms,
                        Workspace &workspace,
                        tree_hmm::CategoricalInputUpdate update) {
  const tree_hmm::PartitionView result =
      Run(model, *workspace.impl_, true, uniforms, update);
  return {
      {workspace.impl_->host_assignments, model.batch * model.plan.num_nodes()},
      result.timings};
}

tree_hmm::BatchedMarginalView
PosteriorMarginalsPrepared(tree_hmm::BatchedModelView model,
                           Workspace &workspace) {
  return RunMarginals(model, *workspace.impl_);
}

tree_hmm::BatchedMarginalView
PosteriorMarginalsPrepared(tree_hmm::BatchedCategoricalModelView model,
                           Workspace &workspace,
                           tree_hmm::CategoricalInputUpdate update) {
  return RunMarginals(model, *workspace.impl_, update);
}

} // namespace tree_hmm::TREE_HMM_GPU_BACKEND_NAMESPACE
