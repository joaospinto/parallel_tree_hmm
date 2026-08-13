#ifndef TREE_HMM_SCALAR_H_
#define TREE_HMM_SCALAR_H_

namespace tree_hmm {

#ifdef TREE_HMM_USE_FLOAT
using Scalar = float;
inline constexpr const char *kPrecisionName = "FP32";
#else
using Scalar = double;
inline constexpr const char *kPrecisionName = "FP64";
#endif

} // namespace tree_hmm

#endif // TREE_HMM_SCALAR_H_
