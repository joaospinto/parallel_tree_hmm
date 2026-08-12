#ifndef TREE_HMM_SRC_METAL_KERNELS_H_
#define TREE_HMM_SRC_METAL_KERNELS_H_

namespace tree_hmm::metal::detail {

inline constexpr char kKernelSource[] = R"metal(
#include <metal_stdlib>
using namespace metal;

struct Rake {
  uint edge;
  uint parent;
  uint leaf;
  uint branch;
};

struct BranchCombination {
  uint destination;
  uint source;
  uint tape;
};

struct BranchAbsorption {
  uint parent;
  uint branch;
  uint tape;
};

struct Compression {
  uint middle;
  uint left_edge;
  uint right_edge;
  uint parent;
  uint child;
  uint tape;
};

struct Params {
  uint states;
  uint nodes;
  uint edges;
  uint branches;
  uint batch;
  uint root;
  uint operation_offset;
  uint operation_count;
  uint scaled;
  uint paths_batched;
};

kernel void initialize_nodes(
    device const float *input [[buffer(0)]],
    device float *nodes [[buffer(1)]],
    constant Params &p [[buffer(2)]],
    uint index [[thread_position_in_grid]]) {
  const uint count = p.batch * p.nodes * p.states;
  if (index < count)
    nodes[index] = input[index];
}

kernel void initialize_paths(
    device const float *input [[buffer(0)]],
    device float *paths [[buffer(1)]],
    constant Params &p [[buffer(2)]],
    uint index [[thread_position_in_grid]]) {
  const uint per_batch = p.edges * p.states * p.states;
  const uint path_batches = p.paths_batched ? p.batch : 1;
  const uint count = path_batches * per_batch;
  if (index < count)
    paths[index] = input[index % per_batch];
}

kernel void rake(
    device const Rake *operations [[buffer(0)]],
    device const float *nodes [[buffer(1)]],
    device const float *paths [[buffer(2)]],
    device float *branches [[buffer(3)]],
    device const float *node_scales [[buffer(4)]],
    device const float *path_scales [[buffer(5)]],
    device float *branch_scales [[buffer(6)]],
    constant Params &p [[buffer(7)]],
    uint3 gid [[thread_position_in_grid]]) {
  threadgroup float normalizer;
  if (gid.x >= p.states || gid.y >= p.operation_count || gid.z >= p.batch)
    return;
  const Rake op = operations[p.operation_offset + gid.y];
  const uint matrix_size = p.states * p.states;
  const uint node_base = (gid.z * p.nodes + op.leaf) * p.states;
  const uint path_batch = p.paths_batched ? gid.z : 0;
  const uint path_base = (path_batch * p.edges + op.edge) * matrix_size;
  const uint branch_base = (gid.z * p.branches + op.branch) * p.states;
  float value = 0.0f;
  for (uint child_state = 0; child_state < p.states; ++child_state) {
    value += paths[path_base + gid.x * p.states + child_state] *
             nodes[node_base + child_state];
  }
  branches[branch_base + gid.x] = value;
  if (!p.scaled)
    return;
  threadgroup_barrier(mem_flags::mem_device);
  if (gid.x == 0) {
    float maximum = 0.0f;
    for (uint state = 0; state < p.states; ++state)
      maximum = max(maximum, branches[branch_base + state]);
    normalizer = maximum > 0.0f ? maximum : 1.0f;
    const float input_scale =
        node_scales[gid.z * p.nodes + op.leaf] +
        path_scales[gid.z * p.edges + op.edge];
    branch_scales[gid.z * p.branches + op.branch] =
        maximum > 0.0f ? input_scale + log(maximum) : -INFINITY;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  branches[branch_base + gid.x] = value / normalizer;
}

// Batched small-state inference has enough independent operations that one
// thread can evaluate and normalize a complete message. This fills SIMD
// groups instead of launching one mostly empty threadgroup per operation.
kernel void rake_serial(
    device const Rake *operations [[buffer(0)]],
    device const float *nodes [[buffer(1)]],
    device const float *paths [[buffer(2)]],
    device float *branches [[buffer(3)]],
    device const float *node_scales [[buffer(4)]],
    device const float *path_scales [[buffer(5)]],
    device float *branch_scales [[buffer(6)]],
    constant Params &p [[buffer(7)]],
    uint index [[thread_position_in_grid]]) {
  const uint count = p.operation_count * p.batch;
  if (index >= count)
    return;
  const uint operation_index = index % p.operation_count;
  const uint batch_index = index / p.operation_count;
  const Rake op = operations[p.operation_offset + operation_index];
  const uint matrix_size = p.states * p.states;
  const uint node_base = (batch_index * p.nodes + op.leaf) * p.states;
  const uint path_batch = p.paths_batched ? batch_index : 0;
  const uint path_base = (path_batch * p.edges + op.edge) * matrix_size;
  const uint branch_base =
      (batch_index * p.branches + op.branch) * p.states;
  float maximum = 0.0f;
  for (uint parent_state = 0; parent_state < p.states; ++parent_state) {
    float value = 0.0f;
    for (uint child_state = 0; child_state < p.states; ++child_state) {
      value += paths[path_base + parent_state * p.states + child_state] *
               nodes[node_base + child_state];
    }
    branches[branch_base + parent_state] = value;
    maximum = max(maximum, value);
  }
  if (!p.scaled)
    return;
  const float normalizer = maximum > 0.0f ? maximum : 1.0f;
  for (uint state = 0; state < p.states; ++state)
    branches[branch_base + state] /= normalizer;
  const float input_scale =
      node_scales[batch_index * p.nodes + op.leaf] +
      path_scales[batch_index * p.edges + op.edge];
  branch_scales[batch_index * p.branches + op.branch] =
      maximum > 0.0f ? input_scale + log(maximum) : -INFINITY;
}

kernel void combine_branches(
    device const BranchCombination *operations [[buffer(0)]],
    device float *branches [[buffer(1)]],
    device float *branch_scales [[buffer(2)]],
    constant Params &p [[buffer(3)]],
    uint3 gid [[thread_position_in_grid]]) {
  threadgroup float normalizer;
  if (gid.x >= p.states || gid.y >= p.operation_count || gid.z >= p.batch)
    return;
  const BranchCombination op = operations[p.operation_offset + gid.y];
  const uint destination_base =
      (gid.z * p.branches + op.destination) * p.states;
  const uint source_base = (gid.z * p.branches + op.source) * p.states;
  const float value =
      branches[destination_base + gid.x] * branches[source_base + gid.x];
  branches[destination_base + gid.x] = value;
  if (!p.scaled)
    return;
  threadgroup_barrier(mem_flags::mem_device);
  if (gid.x == 0) {
    float maximum = 0.0f;
    for (uint state = 0; state < p.states; ++state)
      maximum = max(maximum, branches[destination_base + state]);
    normalizer = maximum > 0.0f ? maximum : 1.0f;
    const uint destination_scale = gid.z * p.branches + op.destination;
    const uint source_scale = gid.z * p.branches + op.source;
    const float input_scale =
        branch_scales[destination_scale] + branch_scales[source_scale];
    branch_scales[destination_scale] =
        maximum > 0.0f ? input_scale + log(maximum) : -INFINITY;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  branches[destination_base + gid.x] = value / normalizer;
}

kernel void combine_branches_serial(
    device const BranchCombination *operations [[buffer(0)]],
    device float *branches [[buffer(1)]],
    device float *branch_scales [[buffer(2)]],
    constant Params &p [[buffer(3)]],
    uint index [[thread_position_in_grid]]) {
  const uint count = p.operation_count * p.batch;
  if (index >= count)
    return;
  const uint operation_index = index % p.operation_count;
  const uint batch_index = index / p.operation_count;
  const BranchCombination op =
      operations[p.operation_offset + operation_index];
  const uint destination_base =
      (batch_index * p.branches + op.destination) * p.states;
  const uint source_base =
      (batch_index * p.branches + op.source) * p.states;
  float maximum = 0.0f;
  for (uint state = 0; state < p.states; ++state) {
    const float value =
        branches[destination_base + state] * branches[source_base + state];
    branches[destination_base + state] = value;
    maximum = max(maximum, value);
  }
  if (!p.scaled)
    return;
  const float normalizer = maximum > 0.0f ? maximum : 1.0f;
  for (uint state = 0; state < p.states; ++state)
    branches[destination_base + state] /= normalizer;
  const uint destination_scale =
      batch_index * p.branches + op.destination;
  const uint source_scale = batch_index * p.branches + op.source;
  const float input_scale =
      branch_scales[destination_scale] + branch_scales[source_scale];
  branch_scales[destination_scale] =
      maximum > 0.0f ? input_scale + log(maximum) : -INFINITY;
}

kernel void absorb_branches(
    device const BranchAbsorption *operations [[buffer(0)]],
    device float *nodes [[buffer(1)]],
    device const float *branches [[buffer(2)]],
    device float *node_scales [[buffer(3)]],
    device const float *branch_scales [[buffer(4)]],
    constant Params &p [[buffer(5)]],
    uint3 gid [[thread_position_in_grid]]) {
  threadgroup float normalizer;
  if (gid.x >= p.states || gid.y >= p.operation_count || gid.z >= p.batch)
    return;
  const BranchAbsorption op = operations[p.operation_offset + gid.y];
  const uint node_base = (gid.z * p.nodes + op.parent) * p.states;
  const uint branch_base = (gid.z * p.branches + op.branch) * p.states;
  const float value = nodes[node_base + gid.x] * branches[branch_base + gid.x];
  nodes[node_base + gid.x] = value;
  if (!p.scaled)
    return;
  threadgroup_barrier(mem_flags::mem_device);
  if (gid.x == 0) {
    float maximum = 0.0f;
    for (uint state = 0; state < p.states; ++state)
      maximum = max(maximum, nodes[node_base + state]);
    normalizer = maximum > 0.0f ? maximum : 1.0f;
    const uint node_scale = gid.z * p.nodes + op.parent;
    const float input_scale =
        node_scales[node_scale] +
        branch_scales[gid.z * p.branches + op.branch];
    node_scales[node_scale] =
        maximum > 0.0f ? input_scale + log(maximum) : -INFINITY;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  nodes[node_base + gid.x] = value / normalizer;
}

kernel void absorb_branches_serial(
    device const BranchAbsorption *operations [[buffer(0)]],
    device float *nodes [[buffer(1)]],
    device const float *branches [[buffer(2)]],
    device float *node_scales [[buffer(3)]],
    device const float *branch_scales [[buffer(4)]],
    constant Params &p [[buffer(5)]],
    uint index [[thread_position_in_grid]]) {
  const uint count = p.operation_count * p.batch;
  if (index >= count)
    return;
  const uint operation_index = index % p.operation_count;
  const uint batch_index = index / p.operation_count;
  const BranchAbsorption op =
      operations[p.operation_offset + operation_index];
  const uint node_base = (batch_index * p.nodes + op.parent) * p.states;
  const uint branch_base =
      (batch_index * p.branches + op.branch) * p.states;
  float maximum = 0.0f;
  for (uint state = 0; state < p.states; ++state) {
    const float value =
        nodes[node_base + state] * branches[branch_base + state];
    nodes[node_base + state] = value;
    maximum = max(maximum, value);
  }
  if (!p.scaled)
    return;
  const float normalizer = maximum > 0.0f ? maximum : 1.0f;
  for (uint state = 0; state < p.states; ++state)
    nodes[node_base + state] /= normalizer;
  const uint node_scale = batch_index * p.nodes + op.parent;
  const float input_scale =
      node_scales[node_scale] +
      branch_scales[batch_index * p.branches + op.branch];
  node_scales[node_scale] =
      maximum > 0.0f ? input_scale + log(maximum) : -INFINITY;
}

kernel void compress(
    device const Compression *operations [[buffer(0)]],
    device const float *nodes [[buffer(1)]],
    device float *paths [[buffer(2)]],
    device const float *node_scales [[buffer(3)]],
    device float *path_scales [[buffer(4)]],
    constant Params &p [[buffer(5)]],
    threadgroup float *scratch [[threadgroup(0)]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint2 group [[threadgroup_position_in_grid]]) {
  if (group.x >= p.operation_count || group.y >= p.batch)
    return;
  const Compression op = operations[p.operation_offset + group.x];
  const uint matrix_size = p.states * p.states;
  threadgroup float *left = scratch;
  threadgroup float *middle = left + matrix_size;
  threadgroup float *right = middle + p.states;
  threadgroup float *normalizer = right + matrix_size;
  const uint left_base = (group.y * p.edges + op.left_edge) * matrix_size;
  const uint right_base = (group.y * p.edges + op.right_edge) * matrix_size;
  const uint middle_base = (group.y * p.nodes + op.middle) * p.states;
  if (thread_index < matrix_size) {
    left[thread_index] = paths[left_base + thread_index];
    right[thread_index] = paths[right_base + thread_index];
  }
  if (thread_index < p.states)
    middle[thread_index] = nodes[middle_base + thread_index];
  threadgroup_barrier(mem_flags::mem_threadgroup);

  const uint parent_state = thread_index / p.states;
  const uint child_state = thread_index % p.states;
  float value = 0.0f;
  for (uint middle_state = 0; middle_state < p.states; ++middle_state) {
    value += left[parent_state * p.states + middle_state] *
             middle[middle_state] *
             right[middle_state * p.states + child_state];
  }
  paths[left_base + thread_index] = value;
  if (!p.scaled)
    return;
  threadgroup_barrier(mem_flags::mem_device);
  if (thread_index == 0) {
    float maximum = 0.0f;
    for (uint entry = 0; entry < matrix_size; ++entry)
      maximum = max(maximum, paths[left_base + entry]);
    *normalizer = maximum > 0.0f ? maximum : 1.0f;
    const uint left_scale = group.y * p.edges + op.left_edge;
    const float input_scale =
        path_scales[left_scale] +
        node_scales[group.y * p.nodes + op.middle] +
        path_scales[group.y * p.edges + op.right_edge];
    path_scales[left_scale] =
        maximum > 0.0f ? input_scale + log(maximum) : -INFINITY;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  paths[left_base + thread_index] = value / *normalizer;
}

kernel void compress_serial4(
    device const Compression *operations [[buffer(0)]],
    device const float *nodes [[buffer(1)]],
    device float *paths [[buffer(2)]],
    device const float *node_scales [[buffer(3)]],
    device float *path_scales [[buffer(4)]],
    constant Params &p [[buffer(5)]],
    uint index [[thread_position_in_grid]]) {
  const uint count = p.operation_count * p.batch;
  if (index >= count)
    return;
  const uint operation_index = index % p.operation_count;
  const uint batch_index = index / p.operation_count;
  const Compression op =
      operations[p.operation_offset + operation_index];
  constexpr uint states = 4;
  constexpr uint matrix_size = states * states;
  const uint left_base =
      (batch_index * p.edges + op.left_edge) * matrix_size;
  const uint right_base =
      (batch_index * p.edges + op.right_edge) * matrix_size;
  const uint middle_base = (batch_index * p.nodes + op.middle) * states;
  float left[matrix_size];
  float right[matrix_size];
  float middle[states];
  for (uint entry = 0; entry < matrix_size; ++entry) {
    left[entry] = paths[left_base + entry];
    right[entry] = paths[right_base + entry];
  }
  for (uint state = 0; state < states; ++state)
    middle[state] = nodes[middle_base + state];

  float maximum = 0.0f;
  for (uint parent_state = 0; parent_state < states; ++parent_state) {
    for (uint child_state = 0; child_state < states; ++child_state) {
      float value = 0.0f;
      for (uint middle_state = 0; middle_state < states; ++middle_state) {
        value += left[parent_state * states + middle_state] *
                 middle[middle_state] *
                 right[middle_state * states + child_state];
      }
      paths[left_base + parent_state * states + child_state] = value;
      maximum = max(maximum, value);
    }
  }
  if (!p.scaled)
    return;
  const float normalizer = maximum > 0.0f ? maximum : 1.0f;
  for (uint entry = 0; entry < matrix_size; ++entry)
    paths[left_base + entry] /= normalizer;
  const uint left_scale = batch_index * p.edges + op.left_edge;
  const float input_scale =
      path_scales[left_scale] +
      node_scales[batch_index * p.nodes + op.middle] +
      path_scales[batch_index * p.edges + op.right_edge];
  path_scales[left_scale] =
      maximum > 0.0f ? input_scale + log(maximum) : -INFINITY;
}

kernel void finish_root(
    device const float *nodes [[buffer(0)]],
    device const float *node_scales [[buffer(1)]],
    device float *output [[buffer(2)]],
    constant Params &p [[buffer(3)]],
    uint batch_index [[thread_position_in_grid]]) {
  if (batch_index >= p.batch)
    return;
  const uint root_base = (batch_index * p.nodes + p.root) * p.states;
  float value = 0.0f;
  for (uint state = 0; state < p.states; ++state)
    value += nodes[root_base + state];
  if (p.scaled) {
    output[batch_index] =
        value > 0.0f
            ? node_scales[batch_index * p.nodes + p.root] + log(value)
            : -INFINITY;
  } else {
    output[batch_index] = value;
  }
}
)metal";

} // namespace tree_hmm::metal::detail

#endif // TREE_HMM_SRC_METAL_KERNELS_H_
