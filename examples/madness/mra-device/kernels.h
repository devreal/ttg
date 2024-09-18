#ifndef HAVE_KERNELS_H
#define HAVE_KERNELS_H

#include <cstddef>
#include "gaussian.h"
#include "tensorview.h"
#include "key.h"
#include "domain.h"
#include "types.h"
#include "ttg/device/device.h"

/* Returns the total size of temporary memory needed for
 * the project() kernel. */
template<mra::Dimension NDIM>
std::size_t project_tmp_size(std::size_t K) {
  const size_t K2NDIM = std::pow(K,NDIM);
  const std::size_t TWOK2NDIM = std::pow(2*K, NDIM);
  return (3*TWOK2NDIM) // workspace, values and r
       + (NDIM*K2NDIM) // xvec in fcube
       + (NDIM*K)      // x in fcube
       + (3*K2NDIM);   // workspace in transform, child_values, r
}

/* Explicitly instantiated for 1, 2, 3 dimensional Gaussians */
template<typename Fn, typename T, mra::Dimension NDIM>
void submit_fcoeffs_kernel(
  const mra::Domain<NDIM>& D,
  const T* gldata,
  const Fn& fn,
  const mra::Key<NDIM>& key,
  mra::TensorView<T, NDIM>& coeffs_view,
  const mra::TensorView<T, 2>& phibar_view,
  const mra::TensorView<T, 2>& hgT_view,
  T* tmp,
  bool* is_leaf_scratch,
  T thresh,
  ttg::device::Stream stream);

template<mra::Dimension NDIM>
std::size_t compress_tmp_size(std::size_t K) {
  const size_t TWOK2NDIM = std::pow(2*K,NDIM);
  const size_t K2NDIM = std::pow(K,NDIM);
  return (TWOK2NDIM) // s
          + K2NDIM // workspace
          + mra::Key<NDIM>::num_children() // sumsq for each child and result
          ;
}


/* Explicitly instantiated for 3D */
template<typename T, mra::Dimension NDIM>
void submit_compress_kernel(
  const mra::Key<NDIM>& key,
  mra::TensorView<T, NDIM>& p_view,
  mra::TensorView<T, NDIM>& result_view,
  const mra::TensorView<T, 2>& hgT_view,
  T* tmp,
  T* sumsqs,
  const std::array<const T*, mra::Key<NDIM>::num_children()>& in_ptrs,
  ttg::device::Stream stream);

template<mra::Dimension NDIM>
std::size_t reconstruct_tmp_size(std::size_t K) {
  const size_t TWOK2NDIM = std::pow(2*K,NDIM); // s & workspace
  return 2*TWOK2NDIM;
}


template<typename T, mra::Dimension NDIM>
void submit_reconstruct_kernel(
  const mra::Key<NDIM>& key,
  mra::TensorView<T, NDIM>& node,
  const mra::TensorView<T, 2>& hg,
  const mra::TensorView<T, NDIM>& from_parent,
  const std::array<T*, mra::Key<NDIM>::num_children()>& r_arr,
  T* tmp,
  ttg::device::Stream stream);

#endif // HAVE_KERNELS_H