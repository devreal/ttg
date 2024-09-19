#include <type_traits>
#include <cstddef>
#include "tensorview.h"
#include "gaussian.h"
#include "key.h"
#include "domain.h"
#include "gl.h"
#include "kernels.h"
#include "functions.h"
#include "util.h"

template<typename T>
struct type_printer;


using namespace mra;

/// Set X(d,mu) to be the mu'th quadrature point in dimension d for the box described by key
template<typename T, Dimension NDIM>
SCOPE void make_quadrature_pts(
  const Domain<NDIM>& D,
  const T* gldata,
  const Key<NDIM>& key,
  TensorView<T,2>& X, std::size_t K)
{
  assert(X.dim(0) == NDIM);
  assert(X.dim(1) == K);
  const Level n = key.level();
  const std::array<Translation,NDIM>& l = key.translation();
  const T h = std::pow(T(0.5),T(n));
  /* retrieve x[] from constant memory, use float */
  const T *x, *w;
  GLget(gldata, &x, &w, K);
  if (threadIdx.z == 0) {
    for (int d = threadIdx.y; d < X.dim(0); d += blockDim.y) {
      T lo, hi; std::tie(lo,hi) = D.get(d);
      T width = h*D.get_width(d);
      for (int i = threadIdx.x; i < X.dim(1); i += blockDim.x) {
        X(d,i) = lo + width*(l[d] + x[i]);
      }
    }
  }
  /* wait for all to complete */
  SYNCTHREADS();
}


//namespace detail {
  template <class functorT> using initial_level_t =
      decltype(std::declval<const functorT>().initial_level());
  template <class functorT> using supports_initial_level =
      ::mra::is_detected<initial_level_t,functorT>;

  template <class functorT, class pairT> using is_negligible_t =
      decltype(std::declval<const functorT>().is_negligible(std::declval<pairT>(),std::declval<double>()));
  template <class functorT, class pairT> using supports_is_negligible =
      ::mra::is_detected<is_negligible_t,functorT,pairT>;
//}


template <typename functionT, typename T, Dimension NDIM>
SCOPE bool is_negligible(
  const functionT& f,
  const std::pair<Coordinate<T,NDIM>, Coordinate<T,NDIM>>& box,
  T thresh)
{
    using pairT = std::pair<Coordinate<T,NDIM>,Coordinate<T,NDIM>>;
    if constexpr (/*detail::*/supports_is_negligible<functionT,pairT>()) return f.is_negligible(box, thresh);
    else return false;
}

/// Make outer product of quadrature points for vectorized algorithms
template<typename T>
SCOPE void make_xvec(const TensorView<T,2>& x, TensorView<T,2>& xvec,
                          std::integral_constant<Dimension, 1>) {
  /* uses threads in 3 dimensions */
  xvec = x;
  /* TensorView assignment synchronizes */
}

/// Make outer product of quadrature points for vectorized algorithms
template<typename T>
SCOPE void make_xvec(const TensorView<T,2>& x, TensorView<T,2>& xvec,
                          std::integral_constant<Dimension, 2>) {
  const std::size_t K = x.dim(1);
  if (threadIdx.z == 0) {
    for (size_t i=threadIdx.y; i<K; i += blockDim.y) {
      for (size_t j=threadIdx.x; j<K; j += blockDim.x) {
        size_t ij = i*K + j;
        xvec(0,ij) = x(0,i);
        xvec(1,ij) = x(1,j);
      }
    }
  }
  SYNCTHREADS();
}

/// Make outer product of quadrature points for vectorized algorithms
template<typename T>
SCOPE void make_xvec(const TensorView<T,2>& x, TensorView<T,2>& xvec,
                          std::integral_constant<Dimension, 3>) {
  const std::size_t K = x.dim(1);
  for (size_t i=threadIdx.z; i<K; i += blockDim.z) {
    for (size_t j=threadIdx.y; j<K; j += blockDim.y) {
      for (size_t k=threadIdx.x; k<K; k += blockDim.x) {
        size_t ijk = i*K*K + j*K + k;
        xvec(0,ijk) = x(0,i);
        xvec(1,ijk) = x(1,j);
        xvec(2,ijk) = x(2,k);
      }
    }
  }
  SYNCTHREADS();
}


template <typename functorT, typename T, Dimension NDIM>
SCOPE
void fcube(const Domain<NDIM>& D,
           const T* gldata,
           const functorT& f,
           const Key<NDIM>& key,
           const T thresh,
           // output
           TensorView<T,3>& values,
           std::size_t K,
           // temporaries
           TensorView<T, 2>& x,
           TensorView<T, 2>& xvec) {
  if (is_negligible(f, D.template bounding_box<T>(key), truncate_tol(key,thresh))) {
      values = 0.0;
      /* TensorView assigment synchronizes */
  }
  else {
    const size_t K = values.dim(0);
    const size_t K2NDIM = std::pow(K,NDIM);
    // sanity checks
    assert(x.dim(0) == NDIM);
    assert(x.dim(1) == K   );
    assert(xvec.dim(0) ==   NDIM);
    assert(xvec.dim(1) == K2NDIM);
    make_quadrature_pts(D, gldata, key, x, K);

    constexpr bool call_coord = std::is_invocable_r<T, decltype(f), Coordinate<T,NDIM>>(); // f(coord)
    constexpr bool call_1d = (NDIM==1) && std::is_invocable_r<T, decltype(f), T>(); // f(x)
    constexpr bool call_2d = (NDIM==2) && std::is_invocable_r<T, decltype(f), T, T>(); // f(x,y)
    constexpr bool call_3d = (NDIM==3) && std::is_invocable_r<T, decltype(f), T, T, T>(); // f(x,y,z)
    constexpr bool call_vec = std::is_invocable<decltype(f), const TensorView<T,2>&, T*, std::size_t>(); // vector API
    static_assert(std::is_invocable<decltype(f), const TensorView<T,2>&, T*, std::size_t>());
    static_assert(call_coord || call_1d || call_2d || call_3d || call_vec, "no working call");

    if constexpr (call_1d || call_2d || call_3d || call_vec) {
      make_xvec(x, xvec, std::integral_constant<Dimension, NDIM>{});
      if constexpr (call_vec) {
        f(xvec, values.data(), K2NDIM);
      }
      else if constexpr (call_1d || call_2d || call_3d) {
        eval_cube_vec(f, xvec, values);
      }
    }
    else if constexpr (call_coord) {
      eval_cube(f, x, values);
    }
    else {
      //throw "how did we get here?";
      // TODO: how to handle this?
      assert(!"Failed to handle eval call!");
    }
    SYNCTHREADS();
  }
}

/* reference implementation, adapted from madness */
template <typename aT, typename bT, typename cT>
SCOPE
void mTxmq(std::size_t dimi, std::size_t dimj, std::size_t dimk,
           cT* __restrict__ c, const aT* a, const bT* b, std::ptrdiff_t ldb=-1) {
  if (ldb == -1) ldb=dimj;
  printf("mTxmq dimi %lu dimj %lu dimk %lu\n", dimi, dimj, dimk);
  /* trivial 2D implementation for devices */
  if (threadIdx.z == 0) {
    for (std::size_t i = threadIdx.y; i < dimi; i += blockDim.y) {
      cT* ci = c + i*dimj; // the row of C all threads in dim x work on
      const aT *aik_ptr = a + i;
      // beta = 0
      for (std::size_t j = threadIdx.x; j < dimj; j += blockDim.x) {
        ci[j] = 0.0;
      }

      for (long k=0; k<dimk; ++k,aik_ptr+=dimi) { /* not parallelized */
        aT aki = *aik_ptr;
        for (std::size_t j = threadIdx.x; j < dimj; j += blockDim.x) {
          ci[j] += aki*b[k*ldb+j];
        }
      }
    }
  }
  SYNCTHREADS();
}
template <Dimension NDIM, typename T>
SCOPE
void transform(const TensorView<T, NDIM>& t,
               const TensorView<T, 2>& c,
               TensorView<T, NDIM>& result,
               TensorView<T, NDIM>& workspace) {
  workspace = 0.0; // set to zero
  const T* pc = c.data();
  T *t0=workspace.data(), *t1=result.data();
  if (t.ndim() & 0x1) std::swap(t0,t1);
  const size_t dimj = c.dim(1);
  size_t dimi = 1;
  for (size_t n=1; n<t.ndim(); ++n) dimi *= dimj;
  mTxmq(dimi, dimj, dimj, t0, t.data(), pc);
  for (size_t n=1; n<t.ndim(); ++n) {
    mTxmq(dimi, dimj, dimj, t1, t0, pc);
    std::swap(t0,t1);
  }
  /* no need to synchronize here, mTxmq synchronizes */
}

template<Dimension NDIM>
SCOPE
std::array<Slice, NDIM> get_child_slice(Key<NDIM> key, std::size_t K, int child) {
  std::array<Slice,NDIM> slices;
  for (size_t d = 0; d < NDIM; ++d) {
    int b = (child>>d) & 0x1;
    slices[d] = Slice(K*b, K*(b+1));
  }
  return slices;
}

template<typename Fn, typename T, Dimension NDIM>
GLOBALSCOPE void fcoeffs_kernel1(
  const Domain<NDIM>& D,
  const T* gldata,
  const Fn& f,
  Key<NDIM> key,
  T* tmp,
  const T* phibar_ptr,
  T* coeffs_ptr,
  const T* hgT_ptr,
  bool *is_leaf,
  std::size_t K,
  T thresh)
{
  int blockid = blockIdx.x;
  bool is_t0 = 0 == (threadIdx.x + threadIdx.y + threadIdx.z);
  const std::size_t K2NDIM = std::pow(K, NDIM);
  const std::size_t TWOK2NDIM = std::pow(2*K, NDIM);
  /* reconstruct tensor views from pointers
   * make sure we have the values at the same offset (0) as in kernel 1 */
  SHARED TensorView<T, NDIM> values, r, child_values, workspace, coeffs;
  SHARED TensorView<T, 2   > hgT, x_vec, x, phibar;
  if (is_t0) {
    values       = TensorView<T, NDIM>(&tmp[0       ], 2*K);
    r            = TensorView<T, NDIM>(&tmp[TWOK2NDIM+0*K2NDIM], K);
    child_values = TensorView<T, NDIM>(&tmp[TWOK2NDIM+1*K2NDIM], K);
    workspace    = TensorView<T, NDIM>(&tmp[TWOK2NDIM+2*K2NDIM], K);
    x_vec        = TensorView<T, 2   >(&tmp[TWOK2NDIM+3*K2NDIM], NDIM, K2NDIM);
    x            = TensorView<T, 2   >(&tmp[TWOK2NDIM+3*K2NDIM + (NDIM*K2NDIM)], NDIM, K);
    phibar       = TensorView<T, 2   >(phibar_ptr, K, K);
  }
  SYNCTHREADS();

  /* compute one child per block */
  for (int bid = blockid; bid < key.num_children(); bid += gridDim.x) {
    Key<NDIM> child = key.child_at(bid);
    auto kl = key.translation();
    auto cl = child.translation();
    //printf("fcoeffs Key (%d, [%d, %d, %d]) child %d (%d, [%d, %d, %d])\n",
    //        key.level(), kl[0], kl[1], kl[2], bid, child.level(), cl[0], cl[1], cl[2]);
    child_values = 0.0; // TODO: needed?
    fcube(D, gldata, f, child, thresh, child_values, K, x, x_vec);
    r = 0.0;
    transform(child_values, phibar, r, workspace);
    auto child_slice = get_child_slice<NDIM>(key, K, bid);
    values(child_slice) = r;
  }

  /* reallocate some of the tensorviews */
  if (is_t0) {
    r          = TensorView<T, NDIM>(&tmp[TWOK2NDIM], 2*K);
    workspace  = TensorView<T, NDIM>(&tmp[2*TWOK2NDIM], 2*K);
    hgT        = TensorView<T, 2>(hgT_ptr, 2*K, 2*K);
    coeffs     = TensorView<T, NDIM>(coeffs_ptr, K);
  }
  SYNCTHREADS();
  r = 0.0;

  T fac = std::sqrt(D.template get_volume<T>()*std::pow(T(0.5),T(NDIM*(1+key.level()))));
  values *= fac;
  // Inlined: filter<T,K,NDIM>(values,r);
  transform<NDIM>(values, hgT, r, workspace);

  auto child_slice = get_child_slice<NDIM>(key, K, 0);
  auto r_slice = r(child_slice);
  coeffs = r_slice; // extract sum coeffs
  r_slice = 0.0; // zero sum coeffs so can easily compute norm of difference coeffs
  /* TensorView assignment synchronizes */
  T norm = mra::normf(r);
  if (is_t0) {
    /* TODO: compute the norm across threads */
    *is_leaf = (norm < truncate_tol(key,thresh)); // test norm of difference coeffs
  }
}

#if 0
template<typename T, Dimension NDIM>
GLOBALSCOPE void fcoeffs_kernel2(
  const Domain<NDIM>& D,
  Key<NDIM> key,
  T* coeffs_ptr,
  const T* hgT_ptr,
  T* tmp,
  bool *is_leaf,
  std::size_t K,
  T thresh)
{
  const bool is_t0 = 0 == (threadIdx.x + threadIdx.y + threadIdx.z);
  const std::size_t    K2NDIM = std::pow(K, NDIM);
  const std::size_t TWOK2NDIM = std::pow(2*K, NDIM);
  /* reconstruct tensor views from pointers
   * make sure we have the values at the same offset (0) as in kernel 1 */
  SHARED TensorView<T, NDIM> values, r, workspace, coeffs;
  SHARED TensorView<T, 2> hgT;
  if (is_t0) {
    values     = TensorView<T, NDIM>(&tmp[0], 2*K);
    r          = TensorView<T, NDIM>(&tmp[TWOK2NDIM], 2*K);
    workspace  = TensorView<T, NDIM>(&tmp[2*TWOK2NDIM], 2*K);
    hgT        = TensorView<T, 2>(hgT_ptr, 2*K, 2*K);
    coeffs     = TensorView<T, NDIM>(coeffs_ptr, K);
  }
  SYNCTHREADS();
  r = 0.0;

  T fac = std::sqrt(D.template get_volume<T>()*std::pow(T(0.5),T(NDIM*(1+key.level()))));
  values *= fac;
  // Inlined: filter<T,K,NDIM>(values,r);
  transform<NDIM>(values, hgT, r, workspace);

  auto child_slice = get_child_slice<NDIM>(key, K, 0);
  auto r_slice = r(child_slice);
  coeffs = r_slice; // extract sum coeffs
  r_slice = 0.0; // zero sum coeffs so can easily compute norm of difference coeffs
  /* TensorView assignment synchronizes */
  T norm = mra::normf(r);
  if (is_t0) {
    /* TODO: compute the norm across threads */
    *is_leaf = (norm < truncate_tol(key,thresh)); // test norm of difference coeffs
  }
}
#endif // 0

template<typename Fn, typename T, Dimension NDIM>
void submit_fcoeffs_kernel(
  const Domain<NDIM>& D,
  const T* gldata,
  const Fn& fn,
  const Key<NDIM>& key,
  TensorView<T, NDIM>& coeffs_view,
  const TensorView<T, 2>& phibar_view,
  const TensorView<T, 2>& hgT_view,
  T* tmp,
  bool* is_leaf_scratch,
  T thresh,
  cudaStream_t stream)
{
  /**
   * Launch two kernels: one with multiple blocks, one with a single block.
   * We use two kernels here because of the synchronization that is required in between
   * (i.e., we have to wait for all children to be computed on before moving to the second part)
   * TODO: We could batch these together into a graph that is launched at once.
   *       Alternatively, we could call the second kernel from the first
   */

  const std::size_t K = coeffs_view.dim(0);
  Dim3 thread_dims = Dim3(K, 1, 1); // figure out how to consider register usage
  /* launch one block per child */
  CALL_KERNEL(fcoeffs_kernel1, 1, thread_dims, 0, stream)(
    D, gldata, fn, key, tmp, phibar_view.data(),
    coeffs_view.data(), hgT_view.data(),
    is_leaf_scratch, K, thresh);
  checkSubmit();
}

/**
 * Instantiate for 3D Gaussian
 */

 template
 void submit_fcoeffs_kernel<Gaussian<double, 3>, double, 3>(
   const Domain<3>& D,
   const double* gldata,
   const Gaussian<double, 3>& fn,
   const Key<3>& key,
   TensorView<double, 3>& coeffs_view,
   const TensorView<double, 2>& phibar_view,
   const TensorView<double, 2>& hgT_view,
   double* tmp,
   bool* is_leaf_scratch,
   double thresh,
   cudaStream_t stream);




/**
 * Compress kernels
 */

template<typename T, Dimension NDIM>
GLOBALSCOPE void compress_kernel(
  Key<NDIM> key,
  T* p_ptr,
  T* result_ptr,
  const T* hgT_ptr,
  T* tmp,
  T* sumsqs, // sumsqs[0]: sum over sumsqs of p; sumsqs[1]: sumsq of result
  const std::array<const T*, Key<NDIM>::num_children()> in_ptrs,
  std::size_t K)
{
  const bool is_t0 = 0 == (threadIdx.x + threadIdx.y + threadIdx.z);
  {   // Collect child coeffs and leaf info
    /* construct tensors */
    const size_t K2NDIM    = std::pow(  K,NDIM);
    const size_t TWOK2NDIM = std::pow(2*K,NDIM);
    SHARED TensorView<T,NDIM> s, r, p, workspace;
    SHARED TensorView<T,2> hgT;
    if (is_t0) {
      s = TensorView<T,NDIM>(&tmp[0], 2*K);
      r = TensorView<T,NDIM>(result_ptr, K);
      p = TensorView<T,NDIM>(p_ptr, K);
      hgT = TensorView<T,2>(hgT_ptr, K);
      workspace = TensorView<T, NDIM>(&tmp[TWOK2NDIM], K);
    }
    SYNCTHREADS();
    r = 0.0;
    p = 0.0;

    for (int bid = blockIdx.x; bid < Key<NDIM>::num_children(); bid += gridDim.x) {
      if (is_t0) sumsqs[bid] = 0.0;
      SYNCTHREADS(); // wait for thread 0 to set sums to 0
      auto child_slice = get_child_slice<NDIM>(key, K, bid);
      const TensorView<T, NDIM> in(in_ptrs[bid], K);
      sumabssq(in, &sumsqs[bid]);
      printf("compress %d sumsq %f\n", bid, sumsqs[bid]);
      s(child_slice) = in;
    }
    //filter<T,K,NDIM>(s,d);  // Apply twoscale transformation=
    transform<NDIM>(s, hgT, r, workspace);
    if (key.level() > 0) {
      auto child_slice = get_child_slice<NDIM>(key, K, 0);
      p = r(child_slice);
      r(child_slice) = 0.0;
    }
    sumabssq(r, &sumsqs[Key<NDIM>::num_children()]); // put result sumsq at the end
  }
}

template<typename T, Dimension NDIM>
void submit_compress_kernel(
  const Key<NDIM>& key,
  TensorView<T, NDIM>& p_view,
  TensorView<T, NDIM>& result_view,
  const TensorView<T, 2>& hgT_view,
  T* tmp,
  T* sumsqs,
  const std::array<const T*, Key<NDIM>::num_children()>& in_ptrs,
  cudaStream_t stream)
{
  const std::size_t K = p_view.dim(0);
  Dim3 thread_dims = Dim3(K, 1, 1); // figure out how to consider register usage

  CALL_KERNEL(compress_kernel, 1, thread_dims, 0, stream)(
    key, p_view.data(), result_view.data(), hgT_view.data(), tmp, sumsqs, in_ptrs, K);
  checkSubmit();
}


/* Instantiations for 3D */
template
void submit_compress_kernel<double, 3>(
  const Key<3>& key,
  TensorView<double, 3>& p_view,
  TensorView<double, 3>& result_view,
  const TensorView<double, 2>& hgT_view,
  double* tmp,
  double* sumsqs,
  const std::array<const double*, Key<3>::num_children()>& in_ptrs,
  cudaStream_t stream);


/**
 * kernel for reconstruct
 */

template<typename T, Dimension NDIM>
GLOBALSCOPE void reconstruct_kernel(
  Key<NDIM> key,
  T* node_ptr,
  T* tmp_ptr,
  const T* hg_ptr,
  const T* from_parent_ptr,
  std::array<T*, (1<<NDIM) > r_arr,
  std::size_t K)
{
  const bool is_t0 = 0 == (threadIdx.x + threadIdx.y + threadIdx.z);
  const size_t TWOK2NDIM = std::pow(2*K,NDIM);
  SHARED TensorView<T, NDIM> node, s, workspace, from_parent;
  SHARED TensorView<T, 2> hg;
  if (is_t0) {
    node        = TensorView<T, NDIM>(node_ptr, 2*K);
    s           = TensorView<T, NDIM>(&tmp_ptr[0], 2*K);
    workspace   = TensorView<T, NDIM>(&tmp_ptr[TWOK2NDIM], 2*K);
    hg          = TensorView<T, 2>(hg_ptr, K);
    from_parent = TensorView<T, NDIM>(from_parent_ptr, K);
  }
  SYNCTHREADS();
  s = 0.0;

  auto child_slice = get_child_slice<NDIM>(key, K, 0);
  if (key.level() != 0) node(child_slice) = from_parent;

  //unfilter<T,K,NDIM>(node.get().coeffs, s);
  transform<NDIM>(node, hg, s, workspace);

  /* extract all r from s
   * NOTE: we could do this on 1<<NDIM blocks but the benefits would likely be small */
  for (int i = 0; i < key.num_children(); ++i) {
    auto child_slice = get_child_slice<NDIM>(key, K, i);
    /* tmp layout: 2K^NDIM for s, K^NDIM for workspace, [K^NDIM]* for r fields */
    auto r = TensorView<T, NDIM>(r_arr[i], K);
    r = s(child_slice);
  }
}

template<typename T, Dimension NDIM>
void submit_reconstruct_kernel(
  const Key<NDIM>& key,
  TensorView<T, NDIM>& node,
  const TensorView<T, 2>& hg,
  const TensorView<T, NDIM>& from_parent,
  const std::array<T*, mra::Key<NDIM>::num_children()>& r_arr,
  T* tmp,
  cudaStream_t stream)
{
  const std::size_t K = node.dim(0);
  /* runs on a single block */
  Dim3 thread_dims = Dim3(K, 1, 1); // figure out how to consider register usage
  CALL_KERNEL(reconstruct_kernel, 1, thread_dims, 0, stream)(
    key, node.data(), tmp, hg.data(), from_parent.data(), r_arr, K);
  checkSubmit();
}


/* explicit instantiation for 3D */
template
void submit_reconstruct_kernel<double, 3>(
  const Key<3>& key,
  TensorView<double, 3>& node,
  const TensorView<double, 2>& hg,
  const TensorView<double, 3>& from_parent,
  const std::array<double*, Key<3>::num_children()>& r_arr,
  double* tmp,
  cudaStream_t stream);