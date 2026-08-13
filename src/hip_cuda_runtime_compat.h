#ifndef TREE_HMM_SRC_HIP_CUDA_RUNTIME_COMPAT_H_
#define TREE_HMM_SRC_HIP_CUDA_RUNTIME_COMPAT_H_

// The numerical backend is written against the small common subset of the
// CUDA and HIP runtime APIs. These aliases keep the implementation shared;
// they do not emulate CUDA facilities or select an NVIDIA execution path.
#define cudaDeviceProp hipDeviceProp_t
#define cudaError_t hipError_t
#define cudaEventCreate hipEventCreate
#define cudaEventDestroy hipEventDestroy
#define cudaEventElapsedTime hipEventElapsedTime
#define cudaEventRecord hipEventRecord
#define cudaEventSynchronize hipEventSynchronize
#define cudaEvent_t hipEvent_t
#define cudaFree hipFree
#define cudaFreeHost hipHostFree
#define cudaGetDeviceCount hipGetDeviceCount
#define cudaGetDeviceProperties hipGetDeviceProperties
#define cudaGetErrorString hipGetErrorString
#define cudaGetLastError hipGetLastError
#define cudaMalloc hipMalloc
#define cudaMallocHost hipHostMalloc
#define cudaMemcpyAsync hipMemcpyAsync
#define cudaMemcpyDeviceToHost hipMemcpyDeviceToHost
#define cudaMemcpyHostToDevice hipMemcpyHostToDevice
#define cudaMemsetAsync hipMemsetAsync
#define cudaSetDevice hipSetDevice
#define cudaStreamCreateWithFlags hipStreamCreateWithFlags
#define cudaStreamDestroy hipStreamDestroy
#define cudaStreamNonBlocking hipStreamNonBlocking
#define cudaStreamSynchronize hipStreamSynchronize
#define cudaStream_t hipStream_t
#define cudaSuccess hipSuccess

#endif // TREE_HMM_SRC_HIP_CUDA_RUNTIME_COMPAT_H_
