#include "tree_hmm/cuda.h"

#include <stdexcept>

namespace tree_hmm::cuda {

struct Workspace::Impl {};

Workspace::Workspace() : impl_(std::make_unique<Impl>()) {}
Workspace::~Workspace() = default;
Workspace::Workspace(Workspace &&) noexcept = default;
Workspace &Workspace::operator=(Workspace &&) noexcept = default;

bool Available() { return false; }

std::string DeviceDescription(int) { return "CUDA backend not built"; }

void Workspace::Reserve(const btrc::Plan &, std::size_t, std::size_t, int) {
  throw std::runtime_error(
      "CUDA backend not built; use Bazel --config=cuda and link tree_hmm_cuda");
}

void Workspace::ReserveCategorical(const btrc::Plan &, std::size_t, std::size_t,
                                   std::size_t, std::span<const btrc::Index>,
                                   int) {
  throw std::runtime_error(
      "CUDA backend not built; use Bazel --config=cuda and link tree_hmm_cuda");
}

void Workspace::ReserveMaximum(const btrc::Plan &, std::size_t, std::size_t,
                               int) {
  throw std::runtime_error(
      "CUDA backend not built; use Bazel --config=cuda and link tree_hmm_cuda");
}

void Workspace::ReserveCategoricalMaximum(const btrc::Plan &, std::size_t,
                                          std::size_t, std::size_t,
                                          std::span<const btrc::Index>, int) {
  throw std::runtime_error(
      "CUDA backend not built; use Bazel --config=cuda and link tree_hmm_cuda");
}

void Workspace::ReserveSumProduct(const btrc::Plan &, std::size_t, std::size_t,
                                  int) {
  throw std::runtime_error(
      "CUDA backend not built; use Bazel --config=cuda and link tree_hmm_cuda");
}

void Workspace::ReserveCategoricalSumProduct(const btrc::Plan &, std::size_t,
                                             std::size_t, std::size_t,
                                             std::span<const btrc::Index>,
                                             int) {
  throw std::runtime_error(
      "CUDA backend not built; use Bazel --config=cuda and link tree_hmm_cuda");
}

tree_hmm::MutableBatchedModelView Workspace::Inputs() {
  throw std::runtime_error(
      "CUDA backend not built; use Bazel --config=cuda and link tree_hmm_cuda");
}

tree_hmm::MutableBatchedModelView Workspace::Inputs(std::size_t) {
  throw std::runtime_error(
      "CUDA backend not built; use Bazel --config=cuda and link tree_hmm_cuda");
}

tree_hmm::MutableBatchedCategoricalModelView Workspace::CategoricalInputs() {
  throw std::runtime_error(
      "CUDA backend not built; use Bazel --config=cuda and link tree_hmm_cuda");
}

tree_hmm::MutableBatchedCategoricalModelView
Workspace::CategoricalInputs(std::size_t) {
  throw std::runtime_error(
      "CUDA backend not built; use Bazel --config=cuda and link tree_hmm_cuda");
}

std::span<float> Workspace::Uniforms() {
  throw std::runtime_error(
      "CUDA backend not built; use Bazel --config=cuda and link tree_hmm_cuda");
}

std::span<float> Workspace::Uniforms(std::size_t) {
  throw std::runtime_error(
      "CUDA backend not built; use Bazel --config=cuda and link tree_hmm_cuda");
}

tree_hmm::PartitionView PartitionFunctionPrepared(tree_hmm::BatchedModelView,
                                                  Workspace &) {
  throw std::runtime_error(
      "CUDA backend not built; use Bazel --config=cuda and link tree_hmm_cuda");
}

tree_hmm::PartitionView LogPartitionFunctionPrepared(tree_hmm::BatchedModelView,
                                                     Workspace &) {
  throw std::runtime_error(
      "CUDA backend not built; use Bazel --config=cuda and link tree_hmm_cuda");
}

tree_hmm::PartitionView
PartitionFunctionPrepared(tree_hmm::BatchedCategoricalModelView, Workspace &) {
  throw std::runtime_error(
      "CUDA backend not built; use Bazel --config=cuda and link tree_hmm_cuda");
}

tree_hmm::PartitionView
LogPartitionFunctionPrepared(tree_hmm::BatchedCategoricalModelView,
                             Workspace &) {
  throw std::runtime_error(
      "CUDA backend not built; use Bazel --config=cuda and link tree_hmm_cuda");
}

tree_hmm::BatchedMaximumAssignmentView
MaximumAPosterioriPrepared(tree_hmm::BatchedModelView, Workspace &) {
  throw std::runtime_error(
      "CUDA backend not built; use Bazel --config=cuda and link tree_hmm_cuda");
}

tree_hmm::BatchedMaximumAssignmentView
MaximumAPosterioriPrepared(tree_hmm::BatchedCategoricalModelView, Workspace &) {
  throw std::runtime_error(
      "CUDA backend not built; use Bazel --config=cuda and link tree_hmm_cuda");
}

tree_hmm::BatchedPosteriorSampleView
PosteriorSamplePrepared(tree_hmm::BatchedModelView, std::span<const float>,
                        Workspace &) {
  throw std::runtime_error(
      "CUDA backend not built; use Bazel --config=cuda and link tree_hmm_cuda");
}

tree_hmm::BatchedPosteriorSampleView
PosteriorSamplePrepared(tree_hmm::BatchedCategoricalModelView,
                        std::span<const float>, Workspace &) {
  throw std::runtime_error(
      "CUDA backend not built; use Bazel --config=cuda and link tree_hmm_cuda");
}

} // namespace tree_hmm::cuda
