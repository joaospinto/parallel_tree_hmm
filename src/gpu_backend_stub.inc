namespace tree_hmm::TREE_HMM_GPU_STUB_NAMESPACE {

struct Workspace::Impl {};

Workspace::Workspace() : impl_(std::make_unique<Impl>()) {}
Workspace::~Workspace() = default;
Workspace::Workspace(Workspace &&) noexcept = default;
Workspace &Workspace::operator=(Workspace &&) noexcept = default;

bool Available() { return false; }

std::string DeviceDescription(int) { return TREE_HMM_GPU_STUB_DESCRIPTION; }

void Workspace::Reserve(const btrc::Plan &, std::size_t, std::size_t, int) {
  throw std::runtime_error(
      TREE_HMM_GPU_STUB_ERROR);
}

void Workspace::ReserveCategorical(const btrc::Plan &, std::size_t, std::size_t,
                                   std::size_t, std::span<const btrc::Index>,
                                   int) {
  throw std::runtime_error(
      TREE_HMM_GPU_STUB_ERROR);
}

void Workspace::ReserveMaximum(const btrc::Plan &, std::size_t, std::size_t,
                               int) {
  throw std::runtime_error(
      TREE_HMM_GPU_STUB_ERROR);
}

void Workspace::ReserveCategoricalMaximum(const btrc::Plan &, std::size_t,
                                          std::size_t, std::size_t,
                                          std::span<const btrc::Index>, int) {
  throw std::runtime_error(
      TREE_HMM_GPU_STUB_ERROR);
}

void Workspace::ReserveSampling(const btrc::Plan &, std::size_t, std::size_t,
                                int) {
  throw std::runtime_error(
      TREE_HMM_GPU_STUB_ERROR);
}

void Workspace::ReserveCategoricalSampling(const btrc::Plan &, std::size_t,
                                           std::size_t, std::size_t,
                                           std::span<const btrc::Index>, int) {
  throw std::runtime_error(
      TREE_HMM_GPU_STUB_ERROR);
}

void Workspace::ReserveMarginals(const btrc::Plan &, std::size_t, std::size_t,
                                 int) {
  throw std::runtime_error(
      TREE_HMM_GPU_STUB_ERROR);
}

void Workspace::ReserveCategoricalMarginals(const btrc::Plan &, std::size_t,
                                            std::size_t, std::size_t,
                                            std::span<const btrc::Index>, int) {
  throw std::runtime_error(
      TREE_HMM_GPU_STUB_ERROR);
}

tree_hmm::MutableBatchedModelView Workspace::Inputs() {
  throw std::runtime_error(
      TREE_HMM_GPU_STUB_ERROR);
}

tree_hmm::MutableBatchedModelView Workspace::Inputs(std::size_t) {
  throw std::runtime_error(
      TREE_HMM_GPU_STUB_ERROR);
}

tree_hmm::MutableBatchedCategoricalModelView Workspace::CategoricalInputs() {
  throw std::runtime_error(
      TREE_HMM_GPU_STUB_ERROR);
}

tree_hmm::MutableBatchedCategoricalModelView
Workspace::CategoricalInputs(std::size_t) {
  throw std::runtime_error(
      TREE_HMM_GPU_STUB_ERROR);
}

std::span<Scalar> Workspace::Uniforms() {
  throw std::runtime_error(
      TREE_HMM_GPU_STUB_ERROR);
}

std::span<Scalar> Workspace::Uniforms(std::size_t) {
  throw std::runtime_error(
      TREE_HMM_GPU_STUB_ERROR);
}

tree_hmm::PartitionView PartitionFunctionPrepared(tree_hmm::BatchedModelView,
                                                  Workspace &) {
  throw std::runtime_error(
      TREE_HMM_GPU_STUB_ERROR);
}

tree_hmm::PartitionView LogPartitionFunctionPrepared(tree_hmm::BatchedModelView,
                                                     Workspace &) {
  throw std::runtime_error(
      TREE_HMM_GPU_STUB_ERROR);
}

tree_hmm::PartitionView
PartitionFunctionPrepared(tree_hmm::BatchedCategoricalModelView, Workspace &,
                          tree_hmm::CategoricalInputUpdate) {
  throw std::runtime_error(
      TREE_HMM_GPU_STUB_ERROR);
}

tree_hmm::PartitionView
LogPartitionFunctionPrepared(tree_hmm::BatchedCategoricalModelView,
                             Workspace &,
                             tree_hmm::CategoricalInputUpdate) {
  throw std::runtime_error(
      TREE_HMM_GPU_STUB_ERROR);
}

tree_hmm::BatchedMaximumAssignmentView
MaximumAPosterioriPrepared(tree_hmm::BatchedModelView, Workspace &) {
  throw std::runtime_error(
      TREE_HMM_GPU_STUB_ERROR);
}

tree_hmm::BatchedMaximumAssignmentView
MaximumAPosterioriPrepared(tree_hmm::BatchedCategoricalModelView, Workspace &,
                           tree_hmm::CategoricalInputUpdate) {
  throw std::runtime_error(
      TREE_HMM_GPU_STUB_ERROR);
}

tree_hmm::BatchedPosteriorSampleView
PosteriorSamplePrepared(tree_hmm::BatchedModelView, std::span<const Scalar>,
                        Workspace &) {
  throw std::runtime_error(
      TREE_HMM_GPU_STUB_ERROR);
}

tree_hmm::BatchedMarginalView
PosteriorMarginalsPrepared(tree_hmm::BatchedModelView, Workspace &) {
  throw std::runtime_error(
      TREE_HMM_GPU_STUB_ERROR);
}

tree_hmm::BatchedPosteriorSampleView
PosteriorSamplePrepared(tree_hmm::BatchedCategoricalModelView,
                        std::span<const Scalar>, Workspace &,
                        tree_hmm::CategoricalInputUpdate) {
  throw std::runtime_error(
      TREE_HMM_GPU_STUB_ERROR);
}

tree_hmm::BatchedMarginalView
PosteriorMarginalsPrepared(tree_hmm::BatchedCategoricalModelView, Workspace &,
                           tree_hmm::CategoricalInputUpdate) {
  throw std::runtime_error(
      TREE_HMM_GPU_STUB_ERROR);
}

} // namespace tree_hmm::TREE_HMM_GPU_STUB_NAMESPACE
