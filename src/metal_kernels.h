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

inline uint node_index(constant Params &p, uint batch, uint node, uint state) {
  return (node * p.batch + batch) * p.states + state;
}

inline uint path_batches(constant Params &p) {
  return p.paths_batched ? p.batch : 1;
}

inline uint path_index(constant Params &p, uint batch, uint edge,
                       uint parent_state, uint child_state) {
  const uint path_batch = p.paths_batched ? batch : 0;
  return ((edge * path_batches(p) + path_batch) * p.states + parent_state) *
             p.states +
         child_state;
}

inline uint branch_index(constant Params &p, uint batch, uint branch,
                         uint state) {
  return (branch * p.batch + batch) * p.states + state;
}

inline uint node_scale_index(constant Params &p, uint batch, uint node) {
  return node * p.batch + batch;
}

inline uint path_scale_index(constant Params &p, uint batch, uint edge) {
  const uint path_batch = p.paths_batched ? batch : 0;
  return edge * path_batches(p) + path_batch;
}

inline uint branch_scale_index(constant Params &p, uint batch, uint branch) {
  return branch * p.batch + batch;
}

kernel void initialize_nodes(
    device const float *input [[buffer(0)]],
    device float *nodes [[buffer(1)]],
    constant Params &p [[buffer(2)]],
    threadgroup float *tile [[threadgroup(0)]],
    uint2 thread_index [[thread_position_in_threadgroup]],
    uint2 group [[threadgroup_position_in_grid]]) {
  constexpr uint tile_size = 32;
  constexpr uint tile_rows = 8;
  constexpr uint tile_stride = tile_size + 1;
  const uint node = group.x * tile_size + thread_index.x;
  const uint batch_base = group.y * tile_size;
  for (uint state = 0; state < p.states; ++state) {
    for (uint row = thread_index.y; row < tile_size; row += tile_rows) {
      const uint batch = batch_base + row;
      if (node < p.nodes && batch < p.batch) {
        tile[row * tile_stride + thread_index.x] =
            input[(batch * p.nodes + node) * p.states + state];
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint batch = batch_base + thread_index.x;
    for (uint row = thread_index.y; row < tile_size; row += tile_rows) {
      const uint output_node = group.x * tile_size + row;
      if (output_node < p.nodes && batch < p.batch) {
        nodes[node_index(p, batch, output_node, state)] =
            tile[thread_index.x * tile_stride + row];
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
}

kernel void initialize_paths(
    device const float *input [[buffer(0)]],
    device float *paths [[buffer(1)]],
    constant Params &p [[buffer(2)]],
    uint index [[thread_position_in_grid]]) {
  const uint batches = path_batches(p);
  const uint count = p.edges * batches;
  if (index >= count)
    return;
  const uint batch = index % batches;
  const uint edge = index / batches;
  const uint matrix_size = p.states * p.states;
  const uint destination = path_index(p, batch, edge, 0, 0);
  const uint source = edge * matrix_size;
  for (uint entry = 0; entry < matrix_size; ++entry)
    paths[destination + entry] = input[source + entry];
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
  const uint node_base = node_index(p, gid.z, op.leaf, 0);
  const uint path_base = path_index(p, gid.z, op.edge, 0, 0);
  const uint branch_base = branch_index(p, gid.z, op.branch, 0);
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
        node_scales[node_scale_index(p, gid.z, op.leaf)] +
        path_scales[path_scale_index(p, gid.z, op.edge)];
    branch_scales[branch_scale_index(p, gid.z, op.branch)] =
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
  const uint batch_index = index % p.batch;
  const uint operation_index = index / p.batch;
  const Rake op = operations[p.operation_offset + operation_index];
  const uint node_base = node_index(p, batch_index, op.leaf, 0);
  const uint path_base = path_index(p, batch_index, op.edge, 0, 0);
  const uint branch_base = branch_index(p, batch_index, op.branch, 0);
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
      node_scales[node_scale_index(p, batch_index, op.leaf)] +
      path_scales[path_scale_index(p, batch_index, op.edge)];
  branch_scales[branch_scale_index(p, batch_index, op.branch)] =
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
  const uint destination_base = branch_index(p, gid.z, op.destination, 0);
  const uint source_base = branch_index(p, gid.z, op.source, 0);
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
    const uint destination_scale =
        branch_scale_index(p, gid.z, op.destination);
    const uint source_scale = branch_scale_index(p, gid.z, op.source);
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
  const uint batch_index = index % p.batch;
  const uint operation_index = index / p.batch;
  const BranchCombination op =
      operations[p.operation_offset + operation_index];
  const uint destination_base =
      branch_index(p, batch_index, op.destination, 0);
  const uint source_base = branch_index(p, batch_index, op.source, 0);
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
      branch_scale_index(p, batch_index, op.destination);
  const uint source_scale = branch_scale_index(p, batch_index, op.source);
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
  const uint node_base = node_index(p, gid.z, op.parent, 0);
  const uint branch_base = branch_index(p, gid.z, op.branch, 0);
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
    const uint node_scale = node_scale_index(p, gid.z, op.parent);
    const float input_scale =
        node_scales[node_scale] +
        branch_scales[branch_scale_index(p, gid.z, op.branch)];
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
  const uint batch_index = index % p.batch;
  const uint operation_index = index / p.batch;
  const BranchAbsorption op =
      operations[p.operation_offset + operation_index];
  const uint node_base = node_index(p, batch_index, op.parent, 0);
  const uint branch_base = branch_index(p, batch_index, op.branch, 0);
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
  const uint node_scale = node_scale_index(p, batch_index, op.parent);
  const float input_scale =
      node_scales[node_scale] +
      branch_scales[branch_scale_index(p, batch_index, op.branch)];
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
  const uint left_base = path_index(p, group.y, op.left_edge, 0, 0);
  const uint right_base = path_index(p, group.y, op.right_edge, 0, 0);
  const uint middle_base = node_index(p, group.y, op.middle, 0);
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
    const uint left_scale = path_scale_index(p, group.y, op.left_edge);
    const float input_scale =
        path_scales[left_scale] +
        node_scales[node_scale_index(p, group.y, op.middle)] +
        path_scales[path_scale_index(p, group.y, op.right_edge)];
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
  const uint batch_index = index % p.batch;
  const uint operation_index = index / p.batch;
  const Compression op =
      operations[p.operation_offset + operation_index];
  constexpr uint states = 4;
  constexpr uint matrix_size = states * states;
  const uint left_base = path_index(p, batch_index, op.left_edge, 0, 0);
  const uint right_base = path_index(p, batch_index, op.right_edge, 0, 0);
  const uint middle_base = node_index(p, batch_index, op.middle, 0);
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
  const uint left_scale = path_scale_index(p, batch_index, op.left_edge);
  const float input_scale =
      path_scales[left_scale] +
      node_scales[node_scale_index(p, batch_index, op.middle)] +
      path_scales[path_scale_index(p, batch_index, op.right_edge)];
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
  const uint root_base = node_index(p, batch_index, p.root, 0);
  float value = 0.0f;
  for (uint state = 0; state < p.states; ++state)
    value += nodes[root_base + state];
  if (p.scaled) {
    output[batch_index] =
        value > 0.0f
            ? node_scales[node_scale_index(p, batch_index, p.root)] + log(value)
            : -INFINITY;
  } else {
    output[batch_index] = value;
  }
}
)metal";

} // namespace tree_hmm::metal::detail

#endif // TREE_HMM_SRC_METAL_KERNELS_H_
