#ifndef MADGL_H_INCL
#define MADGL_H_INCL

#include <cstddef>
#include <cassert>

#ifndef __CUDA_ARCH__
#include "ttg/buffer.h"
#endif // __CUDA_ARCH__

#include "util.h"

namespace mra {

#ifndef __CUDA_ARCH__
  namespace detail {
    /// Arrays for points and weights for the Gauss-Legendre quadrature on [0,1].
    /// only available directly on the host
    extern const double gl_data[128][64];
  } // namespace detail

  /**
   * Host-side functions only
   */

  template<typename T>
  inline ttg::Buffer<const T> GLbuffer() {
    return ttg::Buffer<const T>(&detail::gl_data[0][0], sizeof(detail::gl_data)/sizeof(T));
  }

  template<typename T>
  inline void GLget(const T** x, const T **w, std::size_t N) {
    assert(N>0 && N<=64);
    *x = &detail::gl_data[2*(N-1)  ][0];
    *w = &detail::gl_data[2*(N-1)+1][0];
  }

  /// Evaluate the first k Legendre scaling functions. p should be an array of k elements.
  void legendre_scaling_functions(double x, size_t k, double *p);

  /// Evaluate the first k Legendre scaling functions. p should be an array of k elements.
  void legendre_scaling_functions(float x, size_t k, float *p);

  bool GLinitialize();
#endif // __CUDA_ARCH__

  template<typename T>
  SCOPE void GLget(const T* glptr, const T** x, const T **w, std::size_t N) {
    assert(N>0 && N<=64);
    T (*data)[64] = (T(*)[64])glptr;
    *x = &data[2*(N-1)  ][0];
    *w = &data[2*(N-1)+1][0];
  }

} // namespace mra

#endif
