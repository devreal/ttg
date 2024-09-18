#ifndef MRA_DEVICE_UTIL_H
#define MRA_DEVICE_UTIL_H

namespace mra::detail {
  struct dim3
  {
      unsigned int x, y, z;
      constexpr dim3(unsigned int vx = 1, unsigned int vy = 1, unsigned int vz = 1)
      : x(vx), y(vy), z(vz)
      { }
  };
} // namespace mra::detail

/* convenience macro to mark functions __device__ if compiling for CUDA */
#if defined(__CUDA_ARCH__)
#define SCOPE __device__ __host__
#define VARSCOPE __device__
#define GLOBALSCOPE __global__
#define SYNCTHREADS() __syncthreads()
#define DEVSCOPE __device__
#define SHARED __shared
using Dim3 = dim3;
#else // __CUDA_ARCH__
#define SCOPE
#define VARSCOPE
#define GLOBALSCOPE
#define SYNCTHREADS()
#define DEVSCOPE
#define SHARED
using Dim3 = mra::detail::dim3;
#endif // __CUDA_ARCH__

#ifdef __CUDACC__
#define checkSubmit() \
  if (cudaPeekAtLastError() != cudaSuccess)                         \
    std::cout << "kernel submission failed at " << __LINE__ << ": " \
    << cudaGetErrorString(cudaPeekAtLastError()) << std::endl;
#define CALL_KERNEL(name, ...) name<<<__VA_ARGS__>>>
#else  // __CUDACC__
#define checkSubmit()
#define CALL_KERNEL(name, ...) name
#endif // __CUDACC__

#endif // MRA_DEVICE_UTIL_H