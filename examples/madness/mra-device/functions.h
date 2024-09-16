#ifndef HAVE_MRA_DEVICE_FUNCTIONS_H
#define HAVE_MRA_DEVICE_FUNCTIONS_H

#include "util.h"
#include "types.h"
#include "key.h"
#include "tensorview.h"

namespace mra {


    /// In given box return the truncation tolerance for given threshold
    template <typename T, Dimension NDIM>
    SCOPE T truncate_tol(const Key<NDIM>& key, const T thresh) {
        return thresh; // nothing clever for now
    }

    /// Computes square of distance between two coordinates
    template <typename T>
    SCOPE T distancesq(const Coordinate<T,1>& p, const Coordinate<T,1>& q) {
        T x = p[0]-q[0];
        return x*x;
    }

    template <typename T>
    SCOPE T distancesq(const Coordinate<T,2>& p, const Coordinate<T,2>& q) {
        T x = p[0]-q[0], y = p[1]-q[1];
        return x*x + y*y;
    }

    template <typename T>
    SCOPE T distancesq(const Coordinate<T,3>& p, const Coordinate<T,3>& q) {
        T x = p[0]-q[0], y = p[1]-q[1], z=p[2]-q[2];
        return x*x + y*y + z*z;
    }

    template <typename T>
    SCOPE void distancesq(const Coordinate<T,3>& p, const TensorView<T,1>& q, T* rsq, std::size_t N) {
        const T x = p(0);
#ifdef __CUDA_ARCH__
        int tid = blockDim.x * ((blockDim.y*threadIdx.z) + threadIdx.y) + threadIdx.x;
        for (size_t i = tid; i < N; i += blockDim.x*blockDim.y*blockDim.z) {
            T xx = q(0,i) - x;
            rsq[i] = xx*xx;
        }
#else  // __CUDA_ARCH__
        for (size_t i=0; i<N; i++) {
            T xx = q(0,i) - x;
            rsq[i] = xx*xx;
        }
#endif // __CUDA_ARCH__
    }

    template <typename T>
    SCOPE void distancesq(const Coordinate<T,3>& p, const TensorView<T,2>& q, T* rsq, std::size_t N) {
        const T x = p(0);
        const T y = p(1);
#ifdef __CUDA_ARCH__
        int tid = blockDim.x * ((blockDim.y*threadIdx.z) + threadIdx.y) + threadIdx.x;
        for (size_t i = tid; i < N; i += blockDim.x*blockDim.y*blockDim.z) {
            T xx = q(0,i) - x;
            T yy = q(1,i) - y;
            rsq[i] = xx*xx + yy*yy;
        }
#else  // __CUDA_ARCH__
        for (size_t i=0; i<N; i++) {
            T xx = q(0,i) - x;
            T yy = q(1,i) - y;
            rsq[i] = xx*xx + yy*yy;
        }
#endif // __CUDA_ARCH__
    }

    template <typename T>
    SCOPE void distancesq(const Coordinate<T,3>& p, const TensorView<T,3>& q, T* rsq, std::size_t N) {
        const T x = p(0);
        const T y = p(1);
        const T z = p(2);
#ifdef __CUDA_ARCH__
        int tid = blockDim.x * ((blockDim.y*threadIdx.z) + threadIdx.y) + threadIdx.x;
        for (size_t i = tid; i < N; i += blockDim.x*blockDim.y*blockDim.z) {
            T xx = q(0,i) - x;
            T yy = q(1,i) - y;
            T zz = q(2,i) - z;
            rsq[i] = xx*xx + yy*yy + zz*zz;
        }
#else  // __CUDA_ARCH__
        for (size_t i=0; i<N; i++) {
            T xx = q(0,i) - x;
            T yy = q(1,i) - y;
            T zz = q(2,i) - z;
            rsq[i] = xx*xx + yy*yy + zz*zz;
        }
#endif // __CUDA_ARCH__
    }

    template <typename T, Dimension NDIM, typename accumulatorT>
    SCOPE void sumabssq(const TensorView<T, NDIM>& a, accumulatorT* sum) {
#ifdef __CUDA_ARCH__
      int tid = threadIdx.x + blockIdx.y + blockIdx.z;
      accumulatorT s = 0.0;
      /* play it safe: set sum to zero before the atomic increments */
      if (tid == 0) { *sum = 0.0; }
      /* wait for thread 0 */
      SYNCTHREADS();
      /* every thread computes a partial sum (likely 1 element only) */
      foreach_idx(a, [&](auto... idx) mutable {
        accumulatorT x = a(idx...);
        s += x*x;
      });
      /* accumulate thread-partial results into sum
       * if we had shared memory we could use that here but for now run with atomics
       * NOTE: needs CUDA architecture 6.0 or higher */
      atomicAdd_block(sum, s);
      SYNCTHREADS();
#else  // __CUDA_ARCH__
      accumulatorT s = 0.0;
      foreach_idx(a, [&](auto... idx) mutable {
        accumulatorT x = a(idx...);
        s += x*x;
      });
      *sum = s;
#endif // __CUDA_ARCH__
    }


    /// Compute Frobenius norm ... still needs specializing for complex
    template <typename T, Dimension NDIM, typename accumulatorT = T>
    SCOPE accumulatorT normf(const TensorView<T, NDIM>& a) {
#ifdef __CUDA_ARCH__
      __shared__ accumulatorT sum;
#else  // __CUDA_ARCH__
      accumulatorT sum;
#endif // __CUDA_ARCH__
      sumabssq<accumulatorT>(a, &sum);
#ifdef __CUDA_ARCH__
      /* wait for all threads to contribute */
      SYNCTHREADS();
#endif // __CUDA_ARCH__
      return std::sqrt(sum);
    }

    template<typename T>
    SCOPE void print(const T& t) {
      foreach_idx(t, [&](auto... idx){ printf("[%lu %lu %lu] %f\n", idx..., t(idx...)); });
      SYNCTHREADS();
    }


} // namespace mra


#endif // HAVE_MRA_DEVICE_FUNCTIONS_H