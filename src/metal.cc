#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "tree_hmm/metal.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>

#include "src/metal_kernels.h"

namespace tree_hmm::metal {
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

static_assert(std::is_standard_layout_v<btrc::Rake>);
static_assert(std::is_standard_layout_v<btrc::BranchCombination>);
static_assert(std::is_standard_layout_v<btrc::BranchAbsorption>);
static_assert(std::is_standard_layout_v<btrc::Compression>);
static_assert(sizeof(btrc::Rake) == 16);
static_assert(sizeof(btrc::BranchCombination) == 12);
static_assert(sizeof(btrc::BranchAbsorption) == 12);
static_assert(sizeof(btrc::Compression) == 24);
static_assert(sizeof(Params) == 40);

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

std::string ErrorText(NSError *error) {
  return error == nil ? "unknown Metal error"
                      : std::string(error.localizedDescription.UTF8String);
}

class Runtime {
public:
  Runtime() {
    device_ = MTLCreateSystemDefaultDevice();
    if (device_ == nil)
      throw std::runtime_error("no Metal device is available");
    queue_ = [device_ newCommandQueue];
    if (queue_ == nil)
      throw std::runtime_error("failed to create the Metal command queue");

    NSError *error = nil;
    NSString *source =
        [[NSString alloc] initWithBytes:detail::kKernelSource
                                 length:sizeof(detail::kKernelSource) - 1
                               encoding:NSUTF8StringEncoding];
    id<MTLLibrary> library =
        [device_ newLibraryWithSource:source options:nil error:&error];
    if (library == nil)
      throw std::runtime_error("failed to compile tree-HMM Metal kernels: " +
                               ErrorText(error));
    initialize_nodes_ = Pipeline(library, @"initialize_nodes");
    initialize_paths_ = Pipeline(library, @"initialize_paths");
    rake_ = Pipeline(library, @"rake");
    rake_serial_ = Pipeline(library, @"rake_serial");
    combine_ = Pipeline(library, @"combine_branches");
    combine_serial_ = Pipeline(library, @"combine_branches_serial");
    absorb_ = Pipeline(library, @"absorb_branches");
    absorb_serial_ = Pipeline(library, @"absorb_branches_serial");
    compress_ = Pipeline(library, @"compress");
    compress_serial4_ = Pipeline(library, @"compress_serial4");
    finish_root_ = Pipeline(library, @"finish_root");
  }

  static Runtime &Get() {
    static Runtime runtime;
    return runtime;
  }

  id<MTLDevice> device() const { return device_; }
  id<MTLCommandQueue> queue() const { return queue_; }
  id<MTLComputePipelineState> initialize_nodes() const {
    return initialize_nodes_;
  }
  id<MTLComputePipelineState> initialize_paths() const {
    return initialize_paths_;
  }
  id<MTLComputePipelineState> rake() const { return rake_; }
  id<MTLComputePipelineState> rake_serial() const { return rake_serial_; }
  id<MTLComputePipelineState> combine() const { return combine_; }
  id<MTLComputePipelineState> combine_serial() const { return combine_serial_; }
  id<MTLComputePipelineState> absorb() const { return absorb_; }
  id<MTLComputePipelineState> absorb_serial() const { return absorb_serial_; }
  id<MTLComputePipelineState> compress() const { return compress_; }
  id<MTLComputePipelineState> compress_serial4() const {
    return compress_serial4_;
  }
  id<MTLComputePipelineState> finish_root() const { return finish_root_; }

private:
  id<MTLComputePipelineState> Pipeline(id<MTLLibrary> library, NSString *name) {
    id<MTLFunction> function = [library newFunctionWithName:name];
    if (function == nil)
      throw std::runtime_error("missing Metal function " +
                               std::string(name.UTF8String));
    NSError *error = nil;
    id<MTLComputePipelineState> pipeline =
        [device_ newComputePipelineStateWithFunction:function error:&error];
    if (pipeline == nil)
      throw std::runtime_error("failed to build Metal pipeline " +
                               std::string(name.UTF8String) + ": " +
                               ErrorText(error));
    return pipeline;
  }

  id<MTLDevice> device_;
  id<MTLCommandQueue> queue_;
  id<MTLComputePipelineState> initialize_nodes_;
  id<MTLComputePipelineState> initialize_paths_;
  id<MTLComputePipelineState> rake_;
  id<MTLComputePipelineState> rake_serial_;
  id<MTLComputePipelineState> combine_;
  id<MTLComputePipelineState> combine_serial_;
  id<MTLComputePipelineState> absorb_;
  id<MTLComputePipelineState> absorb_serial_;
  id<MTLComputePipelineState> compress_;
  id<MTLComputePipelineState> compress_serial4_;
  id<MTLComputePipelineState> finish_root_;
};

id<MTLBuffer> MakeBuffer(id<MTLDevice> device, std::size_t bytes) {
  const NSUInteger length =
      static_cast<NSUInteger>(std::max(bytes, sizeof(float)));
  id<MTLBuffer> buffer =
      [device newBufferWithLength:length options:MTLResourceStorageModeShared];
  if (buffer == nil)
    throw std::runtime_error("failed to allocate a Metal buffer");
  return buffer;
}

template <class Value>
id<MTLBuffer> MakeDataBuffer(id<MTLDevice> device,
                             std::span<const Value> values) {
  id<MTLBuffer> buffer =
      MakeBuffer(device, CheckedProduct({values.size(), sizeof(Value)},
                                        "Metal metadata buffer"));
  if (!values.empty())
    std::memcpy(buffer.contents, values.data(), values.size_bytes());
  return buffer;
}

void DispatchOneDimensional(id<MTLComputeCommandEncoder> encoder,
                            id<MTLComputePipelineState> pipeline,
                            NSUInteger count) {
  const NSUInteger width =
      std::min<NSUInteger>(256, pipeline.maxTotalThreadsPerThreadgroup);
  [encoder dispatchThreads:MTLSizeMake(count, 1, 1)
      threadsPerThreadgroup:MTLSizeMake(width, 1, 1)];
}

} // namespace

struct Workspace::Impl {
  const btrc::Plan *plan = nullptr;
  std::size_t states = 0;
  std::size_t batch = 0;
  Params params{};

  id<MTLBuffer> rakes;
  id<MTLBuffer> combinations;
  id<MTLBuffer> absorptions;
  id<MTLBuffer> compressions;
  id<MTLBuffer> input_nodes;
  id<MTLBuffer> input_edges;
  id<MTLBuffer> nodes;
  id<MTLBuffer> paths;
  id<MTLBuffer> branches;
  id<MTLBuffer> node_scales;
  id<MTLBuffer> path_scales;
  id<MTLBuffer> branch_scales;
  id<MTLBuffer> output;
};

bool Available() {
  @autoreleasepool {
    try {
      static_cast<void>(Runtime::Get());
      return true;
    } catch (...) {
      return false;
    }
  }
}

std::string DeviceDescription() {
  @autoreleasepool {
    return std::string(Runtime::Get().device().name.UTF8String);
  }
}

Workspace::Workspace() : impl_(std::make_unique<Impl>()) {}
Workspace::~Workspace() = default;
Workspace::Workspace(Workspace &&) noexcept = default;
Workspace &Workspace::operator=(Workspace &&) noexcept = default;

void Workspace::Reserve(const btrc::Plan &plan, std::size_t states,
                        std::size_t batch) {
  @autoreleasepool {
    if (states == 0 || batch == 0)
      throw std::invalid_argument(
          "Metal state and batch counts must be nonzero");
    Runtime &runtime = Runtime::Get();
    if (CheckedProduct({states, states}, "Metal compression threads") >
        runtime.compress().maxTotalThreadsPerThreadgroup) {
      throw std::invalid_argument(
          "state count exceeds the Metal compression threadgroup limit");
    }

    Impl &storage = *impl_;
    storage.plan = &plan;
    storage.states = states;
    storage.batch = batch;
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

    storage.rakes = MakeDataBuffer(runtime.device(), plan.rakes());
    storage.combinations =
        MakeDataBuffer(runtime.device(), plan.branch_combinations());
    storage.absorptions =
        MakeDataBuffer(runtime.device(), plan.branch_absorptions());
    storage.compressions =
        MakeDataBuffer(runtime.device(), plan.compressions());

    const std::size_t matrix_size = CheckedProduct({states, states}, "matrix");
    storage.input_nodes = MakeBuffer(
        runtime.device(),
        CheckedProduct({batch, plan.num_nodes(), states, sizeof(float)},
                       "Metal input nodes"));
    storage.input_edges = MakeBuffer(
        runtime.device(),
        CheckedProduct({plan.num_edges(), matrix_size, sizeof(float)},
                       "Metal input edges"));
    storage.nodes = MakeBuffer(
        runtime.device(),
        CheckedProduct({batch, plan.num_nodes(), states, sizeof(float)},
                       "Metal node workspace"));
    const std::size_t path_batches = plan.num_compressions() == 0 ? 1 : batch;
    storage.paths = MakeBuffer(runtime.device(),
                               CheckedProduct({path_batches, plan.num_edges(),
                                               matrix_size, sizeof(float)},
                                              "Metal path workspace"));
    storage.branches = MakeBuffer(
        runtime.device(),
        CheckedProduct({batch, plan.num_branches(), states, sizeof(float)},
                       "Metal branch workspace"));
    storage.node_scales =
        MakeBuffer(runtime.device(),
                   CheckedProduct({batch, plan.num_nodes(), sizeof(float)},
                                  "Metal node scales"));
    storage.path_scales =
        MakeBuffer(runtime.device(),
                   CheckedProduct({batch, plan.num_edges(), sizeof(float)},
                                  "Metal path scales"));
    storage.branch_scales =
        MakeBuffer(runtime.device(),
                   CheckedProduct({batch, plan.num_branches(), sizeof(float)},
                                  "Metal branch scales"));
    storage.output =
        MakeBuffer(runtime.device(),
                   CheckedProduct({batch, sizeof(float)}, "Metal output"));
  }
}

tree_hmm::MutableBatchedModelView Workspace::Inputs() {
  return Inputs(impl_->batch);
}

tree_hmm::MutableBatchedModelView Workspace::Inputs(std::size_t batch) {
  @autoreleasepool {
    Impl &storage = *impl_;
    if (storage.plan == nullptr)
      throw std::logic_error("Metal Workspace::Reserve must precede Inputs");
    if (batch == 0 || batch > storage.batch)
      throw std::invalid_argument(
          "Metal input batch exceeds the reserved capacity");
    const std::size_t node_values =
        CheckedProduct({batch, storage.plan->num_nodes(), storage.states},
                       "Metal input nodes");
    const std::size_t edge_values = CheckedProduct(
        {storage.plan->num_edges(), storage.states, storage.states},
        "Metal input edges");
    return {*storage.plan,
            storage.states,
            batch,
            {static_cast<float *>(storage.input_nodes.contents), node_values},
            {static_cast<float *>(storage.input_edges.contents), edge_values}};
  }
}

namespace {

tree_hmm::PartitionView Run(tree_hmm::BatchedModelView model,
                            Workspace::Impl &storage, bool scaled) {
  @autoreleasepool {
    const auto wall_start = Clock::now();
    if (storage.plan != &model.plan || storage.states != model.states ||
        model.batch == 0 || model.batch > storage.batch) {
      throw std::invalid_argument(
          "prepared Metal inference requires Workspace::Reserve for this "
          "plan, state count, and batch capacity");
    }
    const std::size_t expected_nodes =
        CheckedProduct({model.batch, model.plan.num_nodes(), model.states},
                       "Metal node inputs");
    const std::size_t expected_edges =
        CheckedProduct({model.plan.num_edges(), model.states, model.states},
                       "Metal edge inputs");
    if (model.node_potentials.size() != expected_nodes ||
        model.edge_potentials.size() != expected_edges) {
      throw std::invalid_argument(
          "Metal input shapes do not match the workspace");
    }

    const auto upload_start = Clock::now();
    if (model.node_potentials.data() != storage.input_nodes.contents) {
      std::memcpy(storage.input_nodes.contents, model.node_potentials.data(),
                  model.node_potentials.size_bytes());
    }
    if (model.edge_potentials.data() != storage.input_edges.contents) {
      std::memcpy(storage.input_edges.contents, model.edge_potentials.data(),
                  model.edge_potentials.size_bytes());
    }
    const auto upload_end = Clock::now();

    Runtime &runtime = Runtime::Get();
    id<MTLCommandBuffer> command = [runtime.queue() commandBuffer];
    if (command == nil)
      throw std::runtime_error("failed to create a Metal command buffer");

    Params base_params = storage.params;
    base_params.batch = CheckedU32(model.batch, "batch count");
    base_params.scaled = scaled ? 1 : 0;
    if (scaled) {
      id<MTLBlitCommandEncoder> encoder = [command blitCommandEncoder];
      [encoder fillBuffer:storage.node_scales
                    range:NSMakeRange(0, CheckedProduct({model.batch,
                                                         model.plan.num_nodes(),
                                                         sizeof(float)},
                                                        "Metal node scales"))
                    value:0];
      [encoder
          fillBuffer:storage.path_scales
               range:NSMakeRange(
                         0, CheckedProduct(
                                {base_params.paths_batched ? model.batch : 1,
                                 model.plan.num_edges(), sizeof(float)},
                                "Metal path scales"))
               value:0];
      [encoder
          fillBuffer:storage.branch_scales
               range:NSMakeRange(0, CheckedProduct({model.batch,
                                                    model.plan.num_branches(),
                                                    sizeof(float)},
                                                   "Metal branch scales"))
               value:0];
      [encoder endEncoding];
    }

    id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
    if (encoder == nil)
      throw std::runtime_error("failed to create a Metal compute encoder");
    [encoder setComputePipelineState:runtime.initialize_nodes()];
    [encoder setBuffer:storage.input_nodes offset:0 atIndex:0];
    [encoder setBuffer:storage.nodes offset:0 atIndex:1];
    [encoder setBytes:&base_params length:sizeof(Params) atIndex:2];
    DispatchOneDimensional(
        encoder, runtime.initialize_nodes(),
        CheckedProduct({model.batch, model.plan.num_nodes(), model.states},
                       "Metal node initialization"));

    [encoder setComputePipelineState:runtime.initialize_paths()];
    [encoder setBuffer:storage.input_edges offset:0 atIndex:0];
    [encoder setBuffer:storage.paths offset:0 atIndex:1];
    [encoder setBytes:&base_params length:sizeof(Params) atIndex:2];
    DispatchOneDimensional(
        encoder, runtime.initialize_paths(),
        CheckedProduct({base_params.paths_batched ? model.batch : 1,
                        model.plan.num_edges(), model.states, model.states},
                       "Metal path initialization"));

    for (const btrc::PrimitiveBatch &primitive_batch :
         model.plan.primitive_batches()) {
      Params params = base_params;
      params.operation_offset = primitive_batch.offset;
      params.operation_count = primitive_batch.count;
      switch (primitive_batch.primitive) {
      case btrc::Primitive::kRake: {
        const bool serial = model.states <= 8;
        id<MTLComputePipelineState> pipeline =
            serial ? runtime.rake_serial() : runtime.rake();
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:storage.rakes offset:0 atIndex:0];
        [encoder setBuffer:storage.nodes offset:0 atIndex:1];
        [encoder setBuffer:storage.paths offset:0 atIndex:2];
        [encoder setBuffer:storage.branches offset:0 atIndex:3];
        [encoder setBuffer:storage.node_scales offset:0 atIndex:4];
        [encoder setBuffer:storage.path_scales offset:0 atIndex:5];
        [encoder setBuffer:storage.branch_scales offset:0 atIndex:6];
        [encoder setBytes:&params length:sizeof(Params) atIndex:7];
        if (serial) {
          DispatchOneDimensional(
              encoder, pipeline,
              CheckedProduct({model.batch, primitive_batch.count},
                             "Metal rake operations"));
        } else {
          [encoder dispatchThreads:MTLSizeMake(model.states,
                                               primitive_batch.count,
                                               model.batch)
              threadsPerThreadgroup:MTLSizeMake(model.states, 1, 1)];
        }
        break;
      }
      case btrc::Primitive::kBranchCombination: {
        const bool serial = model.states <= 8;
        id<MTLComputePipelineState> pipeline =
            serial ? runtime.combine_serial() : runtime.combine();
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:storage.combinations offset:0 atIndex:0];
        [encoder setBuffer:storage.branches offset:0 atIndex:1];
        [encoder setBuffer:storage.branch_scales offset:0 atIndex:2];
        [encoder setBytes:&params length:sizeof(Params) atIndex:3];
        if (serial) {
          DispatchOneDimensional(
              encoder, pipeline,
              CheckedProduct({model.batch, primitive_batch.count},
                             "Metal branch combinations"));
        } else {
          [encoder dispatchThreads:MTLSizeMake(model.states,
                                               primitive_batch.count,
                                               model.batch)
              threadsPerThreadgroup:MTLSizeMake(model.states, 1, 1)];
        }
        break;
      }
      case btrc::Primitive::kBranchAbsorption: {
        const bool serial = model.states <= 8;
        id<MTLComputePipelineState> pipeline =
            serial ? runtime.absorb_serial() : runtime.absorb();
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:storage.absorptions offset:0 atIndex:0];
        [encoder setBuffer:storage.nodes offset:0 atIndex:1];
        [encoder setBuffer:storage.branches offset:0 atIndex:2];
        [encoder setBuffer:storage.node_scales offset:0 atIndex:3];
        [encoder setBuffer:storage.branch_scales offset:0 atIndex:4];
        [encoder setBytes:&params length:sizeof(Params) atIndex:5];
        if (serial) {
          DispatchOneDimensional(
              encoder, pipeline,
              CheckedProduct({model.batch, primitive_batch.count},
                             "Metal branch absorptions"));
        } else {
          [encoder dispatchThreads:MTLSizeMake(model.states,
                                               primitive_batch.count,
                                               model.batch)
              threadsPerThreadgroup:MTLSizeMake(model.states, 1, 1)];
        }
        break;
      }
      case btrc::Primitive::kCompression: {
        const bool serial = model.states == 4;
        id<MTLComputePipelineState> pipeline =
            serial ? runtime.compress_serial4() : runtime.compress();
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:storage.compressions offset:0 atIndex:0];
        [encoder setBuffer:storage.nodes offset:0 atIndex:1];
        [encoder setBuffer:storage.paths offset:0 atIndex:2];
        [encoder setBuffer:storage.node_scales offset:0 atIndex:3];
        [encoder setBuffer:storage.path_scales offset:0 atIndex:4];
        [encoder setBytes:&params length:sizeof(Params) atIndex:5];
        if (serial) {
          DispatchOneDimensional(
              encoder, pipeline,
              CheckedProduct({model.batch, primitive_batch.count},
                             "Metal compressions"));
        } else {
          const std::size_t matrix_size = model.states * model.states;
          [encoder
              setThreadgroupMemoryLength:(2 * matrix_size + model.states + 1) *
                                         sizeof(float)
                                 atIndex:0];
          [encoder dispatchThreadgroups:MTLSizeMake(primitive_batch.count,
                                                    model.batch, 1)
                  threadsPerThreadgroup:MTLSizeMake(matrix_size, 1, 1)];
        }
        break;
      }
      }
    }

    [encoder setComputePipelineState:runtime.finish_root()];
    [encoder setBuffer:storage.nodes offset:0 atIndex:0];
    [encoder setBuffer:storage.node_scales offset:0 atIndex:1];
    [encoder setBuffer:storage.output offset:0 atIndex:2];
    [encoder setBytes:&base_params length:sizeof(Params) atIndex:3];
    DispatchOneDimensional(encoder, runtime.finish_root(), model.batch);
    [encoder endEncoding];

    [command commit];
    [command waitUntilCompleted];
    if (command.status == MTLCommandBufferStatusError) {
      throw std::runtime_error("Metal tree-HMM execution failed: " +
                               ErrorText(command.error));
    }
    const auto wall_end = Clock::now();
    const double upload_ms =
        std::chrono::duration<double, std::milli>(upload_end - upload_start)
            .count();
    const double wall_ms =
        std::chrono::duration<double, std::milli>(wall_end - wall_start)
            .count();
    const double kernel_ms =
        1000.0 * (command.GPUEndTime - command.GPUStartTime);
    const auto *output = static_cast<const float *>(storage.output.contents);
    return {{output, model.batch}, {upload_ms, kernel_ms, 0.0, wall_ms}};
  }
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

} // namespace tree_hmm::metal
