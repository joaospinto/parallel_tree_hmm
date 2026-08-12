#include "tree_hmm/cuda.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>

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
  Check(cudaMalloc(reinterpret_cast<void **>(&pointer),
                   std::max<std::size_t>(
                       CheckedProduct({count, sizeof(Value)}, "CUDA buffer"),
                       1)),
        "cudaMalloc");
}

template <class Value> void HostAllocate(Value *&pointer, std::size_t count) {
  Check(cudaMallocHost(reinterpret_cast<void **>(&pointer),
                       std::max<std::size_t>(
                           CheckedProduct({count, sizeof(Value)},
                                          "pinned host buffer"),
                           1)),
        "cudaMallocHost");
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
  return (batch * params.nodes + node) * params.states + state;
}

__device__ std::size_t PathIndex(const Params &params, std::size_t batch,
                                 std::size_t edge, std::size_t parent_state,
                                 std::size_t child_state) {
  const std::size_t path_batch = params.paths_batched ? batch : 0;
  return ((path_batch * params.edges + edge) * params.states + parent_state) *
             params.states +
         child_state;
}

__device__ std::size_t BranchIndex(const Params &params, std::size_t batch,
                                   std::size_t branch, std::size_t state) {
  return (batch * params.branches + branch) * params.states + state;
}

__global__ void InitializeNodes(const float *input, float *nodes,
                                std::size_t total) {
  const std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < total)
    nodes[index] = input[index];
}

__global__ void InitializePaths(const float *input, float *paths, Params params,
                                std::size_t total) {
  const std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= total)
    return;
  const std::size_t matrix =
      static_cast<std::size_t>(params.states) * params.states;
  const std::size_t within_batch = index % (params.edges * matrix);
  paths[index] = input[within_batch];
}

__global__ void Rake(const btrc::Rake *operations, const float *nodes,
                     const float *paths, float *branches,
                     const float *node_scales, const float *path_scales,
                     float *branch_scales, Params params) {
  __shared__ float normalizer;
  const std::size_t state = threadIdx.x;
  const std::size_t operation_in_batch = blockIdx.x % params.operation_count;
  const std::size_t batch = blockIdx.x / params.operation_count;
  const btrc::Rake operation =
      operations[params.operation_offset + operation_in_batch];
  if (state >= params.states || batch >= params.batch)
    return;
  const float value = detail::RakeValue(
      paths + PathIndex(params, batch, operation.edge, 0, 0),
      nodes + NodeIndex(params, batch, operation.leaf, 0), params.states,
      state);
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
        node_scales[batch * params.nodes + operation.leaf] +
        path_scales[batch * params.edges + operation.edge];
    branch_scales[batch * params.branches + operation.branch] =
        detail::UpdatedLogScale(input_scale, maximum);
  }
  __syncthreads();
  branches[branch_base + state] = value / normalizer;
}

__global__ void CombineBranches(const btrc::BranchCombination *operations,
                                float *branches, float *branch_scales,
                                Params params) {
  __shared__ float normalizer;
  const std::size_t state = threadIdx.x;
  const std::size_t operation_in_batch = blockIdx.x % params.operation_count;
  const std::size_t batch = blockIdx.x / params.operation_count;
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
        batch * params.branches + operation.destination;
    const std::size_t source_scale =
        batch * params.branches + operation.source;
    const float input_scale =
        branch_scales[destination_scale] + branch_scales[source_scale];
    branch_scales[destination_scale] =
        detail::UpdatedLogScale(input_scale, maximum);
  }
  __syncthreads();
  branches[destination_base + state] = value / normalizer;
}

__global__ void AbsorbBranches(const btrc::BranchAbsorption *operations,
                               float *nodes, const float *branches,
                               float *node_scales,
                               const float *branch_scales, Params params) {
  __shared__ float normalizer;
  const std::size_t state = threadIdx.x;
  const std::size_t operation_in_batch = blockIdx.x % params.operation_count;
  const std::size_t batch = blockIdx.x / params.operation_count;
  const btrc::BranchAbsorption operation =
      operations[params.operation_offset + operation_in_batch];
  if (state >= params.states || batch >= params.batch)
    return;
  const std::size_t node_base =
      NodeIndex(params, batch, operation.parent, 0);
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
    const std::size_t node_scale = batch * params.nodes + operation.parent;
    const float input_scale =
        node_scales[node_scale] +
        branch_scales[batch * params.branches + operation.branch];
    node_scales[node_scale] =
        detail::UpdatedLogScale(input_scale, maximum);
  }
  __syncthreads();
  nodes[node_base + state] = value / normalizer;
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
  const std::size_t operation_in_batch = blockIdx.x % params.operation_count;
  const std::size_t batch = blockIdx.x / params.operation_count;
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
    middle[entry] =
        nodes[NodeIndex(params, batch, operation.middle, entry)];
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
        batch * params.edges + operation.left_edge;
    const float input_scale =
        path_scales[left_scale] +
        node_scales[batch * params.nodes + operation.middle] +
        path_scales[batch * params.edges + operation.right_edge];
    path_scales[left_scale] =
        detail::UpdatedLogScale(input_scale, maximum);
  }
  __syncthreads();
  paths[PathIndex(params, batch, operation.left_edge, parent_state,
                  child_state)] = value / normalizer;
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
                 ? node_scales[batch * params.nodes + params.root] + logf(value)
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
  float *input_nodes = nullptr;
  float *input_edges = nullptr;
  float *nodes = nullptr;
  float *paths = nullptr;
  float *branches = nullptr;
  float *node_scales = nullptr;
  float *path_scales = nullptr;
  float *branch_scales = nullptr;
  float *output = nullptr;

  void Clear() noexcept {
    if (stream != nullptr) {
      static_cast<void>(cudaSetDevice(device));
      static_cast<void>(cudaStreamSynchronize(stream));
    }
    static_cast<void>(cudaFree(rakes));
    static_cast<void>(cudaFree(combinations));
    static_cast<void>(cudaFree(absorptions));
    static_cast<void>(cudaFree(compressions));
    static_cast<void>(cudaFree(input_nodes));
    static_cast<void>(cudaFree(input_edges));
    static_cast<void>(cudaFree(nodes));
    static_cast<void>(cudaFree(paths));
    static_cast<void>(cudaFree(branches));
    static_cast<void>(cudaFree(node_scales));
    static_cast<void>(cudaFree(path_scales));
    static_cast<void>(cudaFree(branch_scales));
    static_cast<void>(cudaFree(output));
    static_cast<void>(cudaFreeHost(host_nodes));
    static_cast<void>(cudaFreeHost(host_edges));
    static_cast<void>(cudaFreeHost(host_output));
    static_cast<void>(cudaEventDestroy(upload_start));
    static_cast<void>(cudaEventDestroy(upload_stop));
    static_cast<void>(cudaEventDestroy(kernel_start));
    static_cast<void>(cudaEventDestroy(kernel_stop));
    static_cast<void>(cudaEventDestroy(download_start));
    static_cast<void>(cudaEventDestroy(download_stop));
    static_cast<void>(cudaStreamDestroy(stream));
    plan = nullptr;
    states = 0;
    batch = 0;
    device = 0;
    params = {};
    stream = nullptr;
    upload_start = nullptr;
    upload_stop = nullptr;
    kernel_start = nullptr;
    kernel_stop = nullptr;
    download_start = nullptr;
    download_stop = nullptr;
    rakes = nullptr;
    combinations = nullptr;
    absorptions = nullptr;
    compressions = nullptr;
    host_nodes = nullptr;
    host_edges = nullptr;
    host_output = nullptr;
    input_nodes = nullptr;
    input_edges = nullptr;
    nodes = nullptr;
    paths = nullptr;
    branches = nullptr;
    node_scales = nullptr;
    path_scales = nullptr;
    branch_scales = nullptr;
    output = nullptr;
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

void Workspace::Reserve(const btrc::Plan &plan, std::size_t states,
                        std::size_t batch, int device) {
  if (states == 0 || batch == 0)
    throw std::invalid_argument("CUDA state and batch counts must be nonzero");
  Impl &storage = *impl_;
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
  HostAllocate(storage.host_nodes, node_values);
  HostAllocate(storage.host_edges, edge_inputs);
  HostAllocate(storage.host_output, batch);
  DeviceAllocate(storage.input_nodes, node_values);
  DeviceAllocate(storage.input_edges, edge_inputs);
  DeviceAllocate(storage.nodes, node_values);
  DeviceAllocate(storage.paths, path_values);
  DeviceAllocate(storage.branches, branch_values);
  DeviceAllocate(storage.node_scales,
                 CheckedProduct({batch, plan.num_nodes()}, "node scales"));
  DeviceAllocate(storage.path_scales,
                 CheckedProduct({batch, plan.num_edges()}, "path scales"));
  DeviceAllocate(
      storage.branch_scales,
      CheckedProduct({batch, plan.num_branches()}, "branch scales"));
  DeviceAllocate(storage.output, batch);
  Check(cudaStreamSynchronize(storage.stream), "topology upload");
}

namespace {

tree_hmm::PartitionView Run(tree_hmm::BatchedModelView model,
                            Workspace::Impl &storage, bool scaled) {
  const auto wall_start = Clock::now();
  if (storage.plan != &model.plan || storage.states != model.states ||
      storage.batch != model.batch) {
    throw std::invalid_argument(
        "prepared CUDA inference requires Workspace::Reserve for this plan, "
        "state count, and batch");
  }
  Check(cudaSetDevice(storage.device), "cudaSetDevice");
  const std::size_t matrix =
      CheckedProduct({model.states, model.states}, "state matrix");
  const std::size_t node_values = CheckedProduct(
      {model.batch, model.plan.num_nodes(), model.states}, "node inputs");
  const std::size_t edge_values =
      CheckedProduct({model.plan.num_edges(), matrix}, "edge inputs");
  if (model.node_potentials.size() != node_values ||
      model.edge_potentials.size() != edge_values) {
    throw std::invalid_argument("CUDA input shapes do not match the workspace");
  }
  std::memcpy(storage.host_nodes, model.node_potentials.data(),
              model.node_potentials.size_bytes());
  std::memcpy(storage.host_edges, model.edge_potentials.data(),
              model.edge_potentials.size_bytes());

  Check(cudaEventRecord(storage.upload_start, storage.stream),
        "cudaEventRecord upload start");
  Check(cudaMemcpyAsync(storage.input_nodes, storage.host_nodes,
                        model.node_potentials.size_bytes(),
                        cudaMemcpyHostToDevice, storage.stream),
        "cudaMemcpyAsync node upload");
  Check(cudaMemcpyAsync(storage.input_edges, storage.host_edges,
                        model.edge_potentials.size_bytes(),
                        cudaMemcpyHostToDevice, storage.stream),
        "cudaMemcpyAsync edge upload");
  Check(cudaEventRecord(storage.upload_stop, storage.stream),
        "cudaEventRecord upload stop");

  constexpr std::size_t kThreads = 256;
  Check(cudaEventRecord(storage.kernel_start, storage.stream),
        "cudaEventRecord kernel start");
  Params base_params = storage.params;
  base_params.scaled = scaled ? 1 : 0;
  if (scaled) {
    Check(cudaMemsetAsync(storage.node_scales, 0,
                          model.batch * model.plan.num_nodes() * sizeof(float),
                          storage.stream),
          "cudaMemsetAsync node scales");
    Check(cudaMemsetAsync(storage.path_scales, 0,
                          model.batch * model.plan.num_edges() * sizeof(float),
                          storage.stream),
          "cudaMemsetAsync path scales");
    Check(cudaMemsetAsync(
              storage.branch_scales, 0,
              model.batch * model.plan.num_branches() * sizeof(float),
              storage.stream),
          "cudaMemsetAsync branch scales");
  }
  InitializeNodes<<<Blocks(node_values, kThreads), kThreads, 0,
                    storage.stream>>>(storage.input_nodes, storage.nodes,
                                      node_values);
  const std::size_t path_values = CheckedProduct(
      {base_params.paths_batched ? model.batch : 1, model.plan.num_edges(),
       matrix},
      "path workspace");
  if (path_values != 0) {
    InitializePaths<<<Blocks(path_values, kThreads), kThreads, 0,
                      storage.stream>>>(storage.input_edges, storage.paths,
                                        base_params, path_values);
  }

  for (const btrc::PrimitiveBatch &batch : model.plan.primitive_batches()) {
    Params params = base_params;
    params.operation_offset = batch.offset;
    params.operation_count = batch.count;
    const std::uint32_t blocks = CheckedU32(
        CheckedProduct({model.batch, batch.count}, "primitive CUDA grid"),
        "primitive CUDA grid");
    switch (batch.primitive) {
    case btrc::Primitive::kRake:
      Rake<<<blocks, model.states, 0, storage.stream>>>(
          storage.rakes, storage.nodes, storage.paths, storage.branches,
          storage.node_scales, storage.path_scales, storage.branch_scales,
          params);
      break;
    case btrc::Primitive::kBranchCombination:
      CombineBranches<<<blocks, model.states, 0, storage.stream>>>(
          storage.combinations, storage.branches, storage.branch_scales,
          params);
      break;
    case btrc::Primitive::kBranchAbsorption:
      AbsorbBranches<<<blocks, model.states, 0, storage.stream>>>(
          storage.absorptions, storage.nodes, storage.branches,
          storage.node_scales, storage.branch_scales, params);
      break;
    case btrc::Primitive::kCompression:
      Compress<<<blocks, matrix, (2 * matrix + model.states) * sizeof(float),
                 storage.stream>>>(storage.compressions, storage.nodes,
                                   storage.paths, storage.node_scales,
                                   storage.path_scales, params);
      break;
    }
  }
  FinishRoot<<<CheckedU32(model.batch, "root CUDA grid"), 1, 0,
               storage.stream>>>(storage.nodes, storage.node_scales,
                                 storage.output, base_params);
  Check(cudaPeekAtLastError(), "tree-HMM CUDA kernel launch");
  Check(cudaEventRecord(storage.kernel_stop, storage.stream),
        "cudaEventRecord kernel stop");

  Check(cudaEventRecord(storage.download_start, storage.stream),
        "cudaEventRecord download start");
  Check(cudaMemcpyAsync(storage.host_output, storage.output,
                        model.batch * sizeof(float), cudaMemcpyDeviceToHost,
                        storage.stream),
        "cudaMemcpyAsync output download");
  Check(cudaEventRecord(storage.download_stop, storage.stream),
        "cudaEventRecord download stop");
  Check(cudaEventSynchronize(storage.download_stop),
        "tree-HMM CUDA execution");

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
  return {{storage.host_output, model.batch},
          {upload_ms, kernel_ms, download_ms, wall_ms}};
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

} // namespace tree_hmm::cuda
