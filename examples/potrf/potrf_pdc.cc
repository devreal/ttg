#define TTG_USE_PARSEC 1

#include <ttg.h>
//#include <madness.h>
#include "../blockmatrix.h"

#include "lapack.hh"

#include <parsec.h>
#include <parsec/data_internal.h>
#include <parsec/data_dist/matrix/matrix.h>
#include <parsec/data_dist/matrix/sym_two_dim_rectangle_cyclic.h>
#include <parsec/data_dist/matrix/two_dim_rectangle_cyclic.h>
// needed for madness::hashT and xterm_debug
#include <madness/world/world.h>

#include <dplasma.h>

#define USE_DPLASMA

//#define PRINT_TILES

static void
dplasma_dprint_tile( int m, int n,
                     const parsec_tiled_matrix_dc_t* descA,
                     const double *M );

/* FLOP macros taken from DPLASMA */
#define FMULS_POTRF(__n) ((double)(__n) * (((1. / 6.) * (double)(__n) + 0.5) * (double)(__n) + (1. / 3.)))
#define FADDS_POTRF(__n) ((double)(__n) * (((1. / 6.) * (double)(__n)      ) * (double)(__n) - (1. / 6.)))
#define FLOPS_DPOTRF(__n) (     FMULS_POTRF((__n)) +       FADDS_POTRF((__n)) )


/* C++ type to PaRSEC's matrix_type mapping */
template<typename T>
struct type2matrixtype
{ };

template<>
struct type2matrixtype<float>
{
    static constexpr const matrix_type value = matrix_type::matrix_RealFloat;
};

template<>
struct type2matrixtype<double>
{
    static constexpr const matrix_type value = matrix_type::matrix_RealDouble;
};


struct Key {
  // ((I, J), K) where (I, J) is the tile coordiante and K is the iteration number
  const int I = 0, J = 0, K = 0;
  madness::hashT hash_val;

  Key() { rehash(); }
  Key(int I, int J, int K) : I(I), J(J), K(K) { rehash(); }

  madness::hashT hash() const { return hash_val; }
  void rehash() {
    hash_val = (static_cast<madness::hashT>(I) << 48)
             ^ (static_cast<madness::hashT>(J) << 32)
             ^ (K << 16);
  }

  // Equality test
  bool operator==(const Key& b) const { return I == b.I && J == b.J && K == b.K; }

  // Inequality test
  bool operator!=(const Key& b) const { return !((*this) == b); }

  template <typename Archive>
  void serialize(Archive& ar) {
    ar& madness::archive::wrap((unsigned char*)this, sizeof(*this));
  }
};

namespace std {
  // specialize std::hash for Key
  template <>
  struct hash<Key> {
    std::size_t operator()(const Key& s) const noexcept { return s.hash(); }
  };
}  // namespace std

std::ostream& operator<<(std::ostream& s, const Key& key) {
  s << "Key(" << key.I << "," << key.J << "," << key.K << ")";
  return s;
}

template<typename T>
class MatrixTile {

public:
  using metadata_t = typename std::pair<int, int>;

  using pointer_t  = typename std::shared_ptr<T>;

private:
  pointer_t _data;
  int _rows, _cols;

  // (Re)allocate the tile memory
  void realloc() {
    //std::cout << "Reallocating new tile" << std::endl;
    _data = std::shared_ptr<T>(new T[_rows * _cols], [](T* p) { delete[] p; });
  }

public:

  MatrixTile(int rows, int cols) : _rows(rows), _cols(cols)
  {
    realloc();
  }

  MatrixTile(const metadata_t& metadata)
  : MatrixTile(std::get<0>(metadata), std::get<1>(metadata))
  { }

  MatrixTile(int rows, int cols, pointer_t data)
  : _data(data), _rows(rows), _cols(cols)
  { }

  MatrixTile(const metadata_t& metadata, pointer_t data)
  : MatrixTile(std::get<0>(metadata), std::get<1>(metadata), std::forward(data))
  { }

  /**
   * Constructor with outside memory. The tile will *not* delete this memory
   * upon destruction.
   */
  MatrixTile(int rows, int cols, T* data)
  : _data(data, [](T*){}), _rows(rows), _cols(cols)
  { }

  MatrixTile(const metadata_t& metadata, T* data)
  : MatrixTile(std::get<0>(metadata), std::get<1>(metadata), data)
  { }


#if 0
  /* Copy dtor and operator with a static_assert to catch unexpected copying */
  MatrixTile(const MatrixTile& other) {
    static_assert("Oops, copy ctor called?!");
  }

  MatrixTile& operator=(const MatrixTile& other) {
    static_assert("Oops, copy ctor called?!");
  }
#endif


  MatrixTile(MatrixTile<T>&& other)  = default;

  MatrixTile& operator=(MatrixTile<T>&& other)  = default;


  /* Defaulted copy ctor and op for shallow copies, see comment below */
  MatrixTile(const MatrixTile<T>& other)  = default;

  MatrixTile& operator=(const MatrixTile<T>& other)  = default;

  /* Deep copy ctor und op are not needed for PO since tiles will never be read
   * and written concurrently. Hence shallow copies are enough, will all
   * receiving tasks sharing tile data. Re-enable this once the PaRSEC backend
   * can handle data sharing without excessive copying */
#if 0
  MatrixTile(const MatrixTile<T>& other)
  : _rows(other._rows), _cols(other._cols)
  {
    this->realloc();
    std::copy_n(other.data(), _rows*_cols, this->data());
  }

  MatrixTile& operator=(const MatrixTile<T>& other) {
    this->_rows = other._rows;
    this->_cols = other._cols;
    this->realloc();
    std::copy_n(other.data(), _rows*_cols, this->data());
  }
#endif // 0

  void set_metadata(metadata_t meta) {
    _rows = std::get<0>(meta);
    _cols = std::get<1>(meta);
  }

  metadata_t get_metadata(void) const {
    return metadata_t{_rows, _cols};
  }

  // Accessing the raw data
  T* data(){
    return _data.get();
  }

  const T* data() const {
    return _data.get();
  }

  size_t size() const {
    return _cols*_rows;
  }

  int rows() const {
    return _rows;
  }

  int cols() const {
    return _cols;
  }
};

namespace ttg {

  template<typename T>
  struct SplitMetadataDescriptor<MatrixTile<T>>
  {

    auto get_metadata(const MatrixTile<T>& t)
    {
      return t.get_metadata();
    }

    auto get_data(MatrixTile<T>& t)
    {
      return std::array<iovec, 1>({t.size()*sizeof(T), t.data()});
    }

    auto create_from_metadata(const typename MatrixTile<T>::metadata_t& meta)
    {
      return MatrixTile<T>(meta);
    }
  };

} // namespace ttg


template<typename PaRSECMatrixT, typename ValueT>
class PaRSECMatrixWrapper {
  PaRSECMatrixT* pm;

public:
  PaRSECMatrixWrapper(PaRSECMatrixT* dc) : pm(dc)
  {
    //std::cout << "PaRSECMatrixWrapper of matrix with " << rows() << "x" << cols() << " tiles " << std::endl;
    //for (int i = 0; i < rows(); ++i) {
    //  for (int j = 0; j <= i; ++j) {
    //    std::cout << "Tile [" << i << ", " << j << "] is at rank " << rank_of(i, j) << std::endl;
    //  }
    //}
  }

  MatrixTile<ValueT> operator()(int row, int col) const {
    ValueT* ptr = static_cast<ValueT*>(parsec_data_copy_get_ptr(
                      parsec_data_get_copy(pm->super.super.data_of(&pm->super.super, row, col), 0)));
    return MatrixTile<ValueT>{pm->super.mb, pm->super.nb, ptr};
  }

  /** Number of tiled rows **/
  int rows(void) const {
    return pm->super.mt;
  }

  /** Number of tiled columns **/
  int cols(void) const {
    return pm->super.nt;
  }

  /* The rank storing the tile at {row, col} */
  int rank_of(int row, int col) const {
    return pm->super.super.rank_of(&pm->super.super, row, col);
  }

  bool is_local(int row, int col) const {
    return ttg::ttg_default_execution_context().rank() == rank_of(row, col);
  }

  PaRSECMatrixT* parsec() {
    return pm;
  }

  const PaRSECMatrixT* parsec() const {
    return pm;
  }

};

template<typename ValueT>
using MatrixT = PaRSECMatrixWrapper<sym_two_dim_block_cyclic_t, ValueT>;


template <typename T>
auto make_potrf(MatrixT<T>& A,
                ttg::Edge<Key, MatrixTile<T>>& input,
                ttg::Edge<Key, MatrixTile<T>>& output_trsm,
                ttg::Edge<Key, MatrixTile<T>>& output_result)
{
  auto f = [=](const Key& key,
               MatrixTile<T>&& tile_kk,
               std::tuple<ttg::Out<Key, MatrixTile<T>>,
                          ttg::Out<Key, MatrixTile<T>>>& out){
    const int I = key.I;
    const int J = key.J;
    const int K = key.K;
    assert(I == J);
    assert(I == K);

#ifdef PRINT_TILES
    std::cout << "POTRF BEFORE:" << std::endl;
    dplasma_dprint_tile(I, I, &A.parsec()->super, tile_kk.data());
#endif // PRINT_TILES

    lapack::potrf(lapack::Uplo::Lower, tile_kk.rows(), tile_kk.data(), tile_kk.rows());

#ifdef PRINT_TILES
    std::cout << "POTRF(" << key << ")" << std::endl;
    std::cout << "POTRF AFTER:" << std::endl;
    dplasma_dprint_tile(I, I, &A.parsec()->super, tile_kk.data());
#endif // PRINT_TILES

    /* tile is done */
    //ttg::send<0>(key, tile_kk, out);

    /* send the tile to outputs */
    std::vector<Key> keylist;
    keylist.reserve(A.rows() - I);
    for (int m = I+1; m < A.rows(); ++m) {
      /* send tile to trsm */
      //std::cout << "POTRF(" << key << "): sending output to " << Key{m, J, K} << std::endl;
      //ttg::send<1>(Key(m, J, K), tile_kk, out);
      keylist.push_back(Key(m, J, K));
    }
    ttg::broadcast<0, 1>(std::make_tuple(std::array<Key, 1>{key}, keylist), std::move(tile_kk), out);
  };
  return ttg::wrap(f, ttg::edges(input), ttg::edges(output_result, output_trsm), "POTRF", {"tile_kk"}, {"output_result", "output_trsm"});
}

template <typename T>
auto make_trsm(MatrixT<T>& A,
               ttg::Edge<Key, MatrixTile<T>>& input_kk,
               ttg::Edge<Key, MatrixTile<T>>& input_mk,
               ttg::Edge<Key, MatrixTile<T>>& output_diag,
               ttg::Edge<Key, MatrixTile<T>>& output_row,
               ttg::Edge<Key, MatrixTile<T>>& output_col,
               ttg::Edge<Key, MatrixTile<T>>& output_result)
{
  auto f = [=](const Key& key,
               const MatrixTile<T>&  tile_kk,
                     MatrixTile<T>&& tile_mk,
                     std::tuple<ttg::Out<Key, MatrixTile<T>>,
                                ttg::Out<Key, MatrixTile<T>>,
                                ttg::Out<Key, MatrixTile<T>>,
                                ttg::Out<Key, MatrixTile<T>>>& out){
    const int I = key.I;
    const int J = key.J;
    const int K = key.K;
    assert(I > K); // we're below (k, k) in row i, column j [k+1 .. NB, k]

    /* No support for different tile sizes yet */
    assert(tile_mk.rows() == tile_kk.rows());
    assert(tile_mk.cols() == tile_kk.cols());

    auto m = tile_mk.rows();

#ifdef PRINT_TILES
    std::cout << "TRSM BEFORE: kk" << std::endl;
    dplasma_dprint_tile(I, J, &A.parsec()->super, tile_kk.data());
    std::cout << "TRSM BEFORE: mk" << std::endl;
    dplasma_dprint_tile(I, J, &A.parsec()->super, tile_mk.data());
#endif // PRINT_TILES

    blas::trsm(blas::Layout::ColMajor,
               blas::Side::Right,
               lapack::Uplo::Lower,
               blas::Op::Trans,
               blas::Diag::NonUnit,
               tile_kk.rows(), m, 1.0,
               tile_kk.data(), m,
               tile_mk.data(), m);

#ifdef PRINT_TILES
    std::cout << "TRSM AFTER: kk" << std::endl;
    dplasma_dprint_tile(I, J, &A.parsec()->super, tile_kk.data());
    std::cout << "TRSM AFTER: mk" << std::endl;
    dplasma_dprint_tile(I, J, &A.parsec()->super, tile_mk.data());
#endif // PRINT_TILES

    //std::cout << "TRSM(" << key << ")" << std::endl;

    std::vector<Key> keylist_row;
    keylist_row.reserve(I-J-1);
    std::vector<Key> keylist_col;
    keylist_row.reserve(A.rows()-I-1);

    /* tile is done */
    //ttg::send<0>(key, std::move(tile_mk), out);

    /* send tile to syrk on diagonal */
    //std::cout << "TRSM(" << key << "): sending output to diag " << Key{I, I, K} << std::endl;
    //ttg::send<1>(Key(I, I, K), tile_mk, out);

    /* send the tile to all gemms across in row i */
    for (int n = J+1; n < I; ++n) {
      //std::cout << "TRSM(" << key << "): sending output to row " << Key{I, n, K} << std::endl;
      //ttg::send<2>(Key(I, n, K), tile_mk, out);
      keylist_row.push_back(Key(I, n, K));
    }

    /* send the tile to all gemms down in column i */
    for (int m = I+1; m < A.rows(); ++m) {
      //std::cout << "TRSM(" << key << "): sending output to col " << Key{m, I, K} << std::endl;
      //ttg::send<3>(Key(m, I, K), tile_mk, out);
      keylist_col.push_back(Key(m, I, K));
    }

    ttg::broadcast<0, 1, 2, 3>(std::make_tuple(std::array<Key, 1>{key},
                                               std::array<Key, 1>{Key(I, I, K)},
                                               keylist_row, keylist_col),
                            std::move(tile_mk), out);
  };
  return ttg::wrap(f, ttg::edges(input_kk, input_mk), ttg::edges(output_result, output_diag, output_row, output_col),
                   "TRSM", {"tile_kk", "tile_mk"}, {"output_result", "output_diag", "output_row", "output_col"});
}


template <typename T>
auto make_syrk(MatrixT<T>& A,
               ttg::Edge<Key, MatrixTile<T>>& input_mk,
               ttg::Edge<Key, MatrixTile<T>>& input_mm,
               ttg::Edge<Key, MatrixTile<T>>& output_potrf,
               ttg::Edge<Key, MatrixTile<T>>& output_syrk)
{
  auto f = [=](const Key& key,
               const MatrixTile<T>&  tile_mk,
                     MatrixTile<T>&& tile_mm,
                     std::tuple<ttg::Out<Key, MatrixTile<T>>,
                                ttg::Out<Key, MatrixTile<T>>>& out){
    const int I = key.I;
    const int J = key.J;
    const int K = key.K;
    assert(I == J);
    assert(I > K);

    /* No support for different tile sizes yet */
    assert(tile_mk.rows() == tile_mm.rows());
    assert(tile_mk.cols() == tile_mm.cols());

    auto m = tile_mk.rows();

#ifdef PRINT_TILES
    std::cout << "SYRK BEFORE: mk" << std::endl;
    dplasma_dprint_tile(I, I, &A.parsec()->super, tile_mk.data());
    std::cout << "SYRK BEFORE: mk" << std::endl;
    dplasma_dprint_tile(I, I, &A.parsec()->super, tile_mm.data());
#endif // PRINT_TILES

    blas::syrk(blas::Layout::ColMajor,
               lapack::Uplo::Lower,
               blas::Op::NoTrans,
               tile_mk.rows(), m, -1.0,
               tile_mk.data(), m, 1.0,
               tile_mm.data(), m);

#ifdef PRINT_TILES
    std::cout << "SYRK AFTER: mk" << std::endl;
    dplasma_dprint_tile(I, I, &A.parsec()->super, tile_mk.data());
    std::cout << "SYRK AFTER: nk" << std::endl;
    dplasma_dprint_tile(I, I, &A.parsec()->super, tile_mm.data());
    std::cout << "SYRK(" << key << ")" << std::endl;
#endif // PRINT_TILES

    if (I == K+1) {
      /* send the tile to potrf */
      //std::cout << "SYRK(" << key << "): sending output to POTRF " << Key{I, I, K+1} << std::endl;
      ttg::send<0>(Key(I, I, K+1), std::move(tile_mm), out);
    } else {
      /* send output to next syrk */
      //std::cout << "SYRK(" << key << "): sending output to SYRK " << Key{I, I, K+1} << std::endl;
      ttg::send<1>(Key(I, I, K+1), std::move(tile_mm), out);
    }

  };
  return ttg::wrap(f,
                   ttg::edges(input_mk, input_mm),
                   ttg::edges(output_potrf, output_syrk), "SYRK",
                   {"tile_mk", "tile_mm"}, {"output_potrf", "output_syrk"});
}


template <typename T>
auto make_gemm(MatrixT<T>& A,
               ttg::Edge<Key, MatrixTile<T>>& input_nk,
               ttg::Edge<Key, MatrixTile<T>>& input_mk,
               ttg::Edge<Key, MatrixTile<T>>& input_nm,
               ttg::Edge<Key, MatrixTile<T>>& output_trsm,
               ttg::Edge<Key, MatrixTile<T>>& output_gemm)
{
  auto f = [=](const Key& key,
              const MatrixTile<T>& tile_nk,
              const MatrixTile<T>& tile_mk,
                    MatrixTile<T>&& tile_nm,
                    std::tuple<ttg::Out<Key, MatrixTile<T>>,
                               ttg::Out<Key, MatrixTile<T>>>& out){
    const int I = key.I;
    const int J = key.J;
    const int K = key.K;
    assert(I != J && I > K && J > K);

    /* No support for different tile sizes yet */
    assert(tile_nk.rows() == tile_mk.rows() && tile_nk.rows() == tile_nm.rows());
    assert(tile_nk.cols() == tile_mk.cols() && tile_nk.cols() == tile_nm.cols());

    auto m = tile_nk.rows();

#ifdef PRINT_TILES
    std::cout << "GEMM BEFORE: nk" << std::endl;
    dplasma_dprint_tile(I, I, &A.parsec()->super, tile_nk.data());
    std::cout << "GEMM BEFORE: mk" << std::endl;
    dplasma_dprint_tile(I, I, &A.parsec()->super, tile_mk.data());
    std::cout << "GEMM BEFORE: nm" << std::endl;
    dplasma_dprint_tile(I, I, &A.parsec()->super, tile_nm.data());
#endif // PRINT_TILES

    blas::gemm(blas::Layout::ColMajor,
               blas::Op::NoTrans,
               blas::Op::Trans,
               m, m, m, -1.0,
               tile_nk.data(), m,
               tile_mk.data(), m, 1.0,
               tile_nm.data(), m);

#ifdef PRINT_TILES
    std::cout << "GEMM AFTER: nk" << std::endl;
    dplasma_dprint_tile(I, I, &A.parsec()->super, tile_nk.data());
    std::cout << "GEMM AFTER: mk" << std::endl;
    dplasma_dprint_tile(I, I, &A.parsec()->super, tile_mk.data());
    std::cout << "GEMM AFTER: nm" << std::endl;
    dplasma_dprint_tile(I, I, &A.parsec()->super, tile_nm.data());
    std::cout << "GEMM(" << key << ")" << std::endl;
#endif // PRINT_TILES

    /* send the tile to output */
    if (J == K+1) {
      /* send the tile to trsm */
      //std::cout << "GEMM(" << key << "): sending output to TRSM " << Key{I, J, K+1} << std::endl;
      ttg::send<0>(Key(I, J, K+1), std::move(tile_nm), out);
    } else {
      /* send the tile to the next gemm */
      //std::cout << "GEMM(" << key << "): sending output to GEMM " << Key{I, J, K+1} << std::endl;
      ttg::send<1>(Key(I, J, K+1), std::move(tile_nm), out);
    }
  };
  return ttg::wrap(f,
                   ttg::edges(input_nk, input_mk, input_nm),
                   ttg::edges(output_trsm, output_gemm), "GEMM",
                   {"input_nk", "input_mk", "input_nm"},
                   {"output_trsm", "outout_gemm"});
}

template<typename T>
auto initiator(MatrixT<T>& A,
               ttg::Edge<Key, MatrixTile<T>>& syrk_potrf,
               ttg::Edge<Key, MatrixTile<T>>& gemm_trsm,
               ttg::Edge<Key, MatrixTile<T>>& syrk_syrk,
               ttg::Edge<Key, MatrixTile<T>>& gemm_gemm)
{
  auto f = [=](const Key& key,
               std::tuple<ttg::Out<Key, MatrixTile<T>>,
                          ttg::Out<Key, MatrixTile<T>>,
                          ttg::Out<Key, MatrixTile<T>>,
                          ttg::Out<Key, MatrixTile<T>>>& out){
    /* kick off first POTRF */
    //std::cout << "Initiator called with " << key << std::endl;
    if (A.is_local(0, 0)) {
      ttg::send<0>(Key{0, 0, 0}, A(0, 0), out);
    }
    for (int i = 1; i < A.rows(); i++) {
      /* send gemm input to TRSM */
      if (A.is_local(i, 0)) {
        //std::cout << "Initiating TRSM " << Key{i, 0, 0} << std::endl;
        ttg::send<1>(Key{i, 0, 0}, A(i, 0), out);
      }
      /* send syrk to SYRK */
      if (A.is_local(i, i)) {
        ttg::send<2>(Key{i, i, 0}, A(i, i), out);
      }
      for (int j = 1; j < i; j++) {
        /* send gemm to GEMM */
        if (A.is_local(i, j)) {
          ttg::send<3>(Key{i, j, 0}, A(i, j), out);
        }
      }
    }
  };

  return ttg::wrap<Key>(f, ttg::edges(), ttg::edges(syrk_potrf, gemm_trsm, syrk_syrk, gemm_gemm), "INITIATOR");
}

template <typename T>
auto make_result(MatrixT<T>& A, const ttg::Edge<Key, MatrixTile<T>>& result) {
  auto f = [=](const Key& key, MatrixTile<T>&& tile, std::tuple<>& out) {
    /* write back any tiles that are not in the matrix already */
    const int I = key.I;
    const int J = key.J;
    if (A(I, J).data() != tile.data()) {
      std::cout << "Writing back tile {" << I << ", " << J << "} " << std::endl;
      std::copy_n(tile.data(), tile.rows()*tile.cols(), A(I, J).data());
    }
  };

  return ttg::wrap(f, ttg::edges(result), ttg::edges(), "Final Output", {"result"}, {});
}


int main(int argc, char **argv)
{

  std::chrono::time_point<std::chrono::high_resolution_clock> beg, end;
  int N = 1024;
  int M = N;
  int NB = 128;
  int check = 0;
  ttg::ttg_initialize(argc, argv, 1);

  auto world = ttg::ttg_default_execution_context();

  if (argc > 1) {
    N = M = atoi(argv[1]);
  }

  if (argc > 2) {
    NB = atoi(argv[2]);
  }

  if (argc > 3) {
    check = atoi(argv[3]);
  }

  int n_rows = (N / NB) + (N % NB > 0);
  int n_cols = (M / NB) + (M % NB > 0);

  int P = std::sqrt(world.size());
  int Q = (world.size() + P - 1)/P;

  static_assert(ttg::has_split_metadata<MatrixTile<double>>::value);

  std::cout << "Creating 2D block cyclic matrix with NB " << NB << " N " << N << " M " << M << " P " << P << std::endl;

  sym_two_dim_block_cyclic_t dcA;
  sym_two_dim_block_cyclic_init(&dcA, matrix_type::matrix_RealDouble,
                                world.size(), world.rank(), NB, NB, N, M,
                                0, 0, N, M, P, matrix_Lower);
  dcA.mat = parsec_data_allocate((size_t)dcA.super.nb_local_tiles *
                                 (size_t)dcA.super.bsiz *
                                 (size_t)parsec_datadist_getsizeoftype(dcA.super.mtype));
  parsec_data_collection_set_key((parsec_data_collection_t*)&dcA, "Matrix A");

  ttg::Edge<Key, MatrixTile<double>> potrf_trsm("potrf_trsm"),
                                      trsm_syrk("trsm_syrk"),
                                      syrk_potrf("syrk_potrf"),
                                      syrk_syrk("syrk_syrk"),
                                      gemm_gemm("gemm_gemm"),
                                      gemm_trsm("gemm_trsm"),
                                      trsm_gemm_row("trsm_gemm_row"),
                                      trsm_gemm_col("trsm_gemm_col"),
                                      result("result");

  //Matrix<double>* A = new Matrix<double>(n_rows, n_cols, NB, NB);
  MatrixT<double> A{&dcA};
  /* TODO: initialize the matrix */
  /* This works only with the parsec backend! */
  int random_seed = 3872;

#ifdef USE_DPLASMA
  dplasma_dplgsy( world.impl().context(), (double)(N), matrix_Lower,
                (parsec_tiled_matrix_dc_t *)&dcA, random_seed);
#endif // USE_DPLASMA

  //dplasma_dprint(world.impl().context(), matrix_Lower, dcA);
  // plgsy(A);

  auto keymap = [&](const Key& key) {
    //std::cout << "Key " << key << " is at rank " << A.rank_of(key.I, key.J) << std::endl;
    return A.rank_of(key.I, key.J);
  };

  auto op_init  = initiator(A, syrk_potrf, gemm_trsm, syrk_syrk, gemm_gemm);
  /* op_init gets a special keymap where all keys are local */
  op_init->set_keymap([&](const Key&){ return world.rank(); });
  auto op_potrf = make_potrf(A, syrk_potrf, potrf_trsm, result);
  op_potrf->set_keymap(keymap);
  auto op_trsm  = make_trsm(A,
                            potrf_trsm, gemm_trsm,
                            trsm_syrk, trsm_gemm_row, trsm_gemm_col, result);
  op_trsm->set_keymap(keymap);
  auto op_syrk  = make_syrk(A, trsm_syrk, syrk_syrk, syrk_potrf, syrk_syrk);
  op_syrk->set_keymap(keymap);
  auto op_gemm  = make_gemm(A,
                            trsm_gemm_row, trsm_gemm_col, gemm_gemm,
                            gemm_trsm, gemm_gemm);
  op_gemm->set_keymap(keymap);
  auto op_result = make_result(A, result);
  op_result->set_keymap(keymap);

  auto connected = make_graph_executable(op_init.get());
  assert(connected);
  TTGUNUSED(connected);
  std::cout << "Graph is connected: " << connected << std::endl;

  if (world.rank() == 0) {
#if 0
    std::cout << "==== begin dot ====\n";
    std::cout << ttg::Dot()(op_init.get()) << std::endl;
    std::cout << "==== end dot ====\n";
#endif // 0
    beg = std::chrono::high_resolution_clock::now();
  }
  op_init->invoke(Key{0, 0, 0});

  ttg::ttg_execute(world);
  ttg::ttg_fence(world);
  if (world.rank() == 0) {
    end = std::chrono::high_resolution_clock::now();
    auto elapsed = (std::chrono::duration_cast<std::chrono::microseconds>(end - beg).count());
    end = std::chrono::high_resolution_clock::now();
    std::cout << "TTG Execution Time (milliseconds) : "
              << elapsed / 1000 << " : Flops " << (FLOPS_DPOTRF(uint64_t(N))/1E9)/(elapsed/1E3) << " GF/s" << std::endl;
  }

#ifdef USE_DPLASMA
  if( check ) {
    /* Check the factorization */
    int loud = 10;
    int ret = 0;
    sym_two_dim_block_cyclic_t dcA0;
    sym_two_dim_block_cyclic_init(&dcA0, matrix_type::matrix_RealDouble,
                                  world.size(), world.rank(), NB, NB, N, M,
                                  0, 0, N, M, P, matrix_Lower);
    dcA0.mat = parsec_data_allocate((size_t)dcA0.super.nb_local_tiles *
                                  (size_t)dcA0.super.bsiz *
                                  (size_t)parsec_datadist_getsizeoftype(dcA0.super.mtype));
    parsec_data_collection_set_key((parsec_data_collection_t*)&dcA0, "Matrix A0");
    dplasma_dplgsy( world.impl().context(), (double)(N), matrix_Lower,
                  (parsec_tiled_matrix_dc_t *)&dcA0, random_seed);

    ret |= check_dpotrf( world.impl().context(), (world.rank() == 0) ? loud : 0, matrix_Lower,
                          (parsec_tiled_matrix_dc_t *)&dcA,
                          (parsec_tiled_matrix_dc_t *)&dcA0);

    /* Check the solution */
    two_dim_block_cyclic_t dcB;
    two_dim_block_cyclic_init(&dcB, matrix_type::matrix_RealDouble, matrix_storage::matrix_Tile,
                              world.size(), world.rank(), NB, NB, N, M,
                              0, 0, N, M, 1, 1, P);
    dcB.mat = parsec_data_allocate((size_t)dcB.super.nb_local_tiles *
                                  (size_t)dcB.super.bsiz *
                                  (size_t)parsec_datadist_getsizeoftype(dcB.super.mtype));
    parsec_data_collection_set_key((parsec_data_collection_t*)&dcB, "Matrix B");
    dplasma_dplrnt( world.impl().context(), 0, (parsec_tiled_matrix_dc_t *)&dcB, random_seed+1);

    two_dim_block_cyclic_t dcX;
    two_dim_block_cyclic_init(&dcX, matrix_type::matrix_RealDouble, matrix_storage::matrix_Tile,
                              world.size(), world.rank(), NB, NB, N, M,
                              0, 0, N, M, 1, 1, P);
    dcX.mat = parsec_data_allocate((size_t)dcX.super.nb_local_tiles *
                                  (size_t)dcX.super.bsiz *
                                  (size_t)parsec_datadist_getsizeoftype(dcX.super.mtype));
    parsec_data_collection_set_key((parsec_data_collection_t*)&dcX, "Matrix X");
    dplasma_dlacpy( world.impl().context(), dplasmaUpperLower,
                    (parsec_tiled_matrix_dc_t *)&dcB, (parsec_tiled_matrix_dc_t *)&dcX );

    dplasma_dpotrs(world.impl().context(), matrix_Lower,
                    (parsec_tiled_matrix_dc_t *)&dcA,
                    (parsec_tiled_matrix_dc_t *)&dcX );

    ret |= check_daxmb( world.impl().context(), (world.rank() == 0) ? loud : 0, matrix_Lower,
                        (parsec_tiled_matrix_dc_t *)&dcA0,
                        (parsec_tiled_matrix_dc_t *)&dcB,
                        (parsec_tiled_matrix_dc_t *)&dcX);

    /* Cleanup */
    parsec_data_free(dcA0.mat); dcA0.mat = NULL;
    parsec_tiled_matrix_dc_destroy( (parsec_tiled_matrix_dc_t*)&dcA0 );
    parsec_data_free(dcB.mat); dcB.mat = NULL;
    parsec_tiled_matrix_dc_destroy( (parsec_tiled_matrix_dc_t*)&dcB );
    parsec_data_free(dcX.mat); dcX.mat = NULL;
    parsec_tiled_matrix_dc_destroy( (parsec_tiled_matrix_dc_t*)&dcX );
  }
#endif // USE_DPLASMA

  //delete A;
  /* cleanup allocated matrix before shutting down PaRSEC */
  parsec_data_free(dcA.mat); dcA.mat = NULL;
  parsec_tiled_matrix_dc_destroy( (parsec_tiled_matrix_dc_t*)&dcA);

  ttg::ttg_finalize();
  return 0;
}


/**
 *******************************************************************************
 *
 * @ingroup dplasma_double_check
 *
 * check_dpotrf - Check the correctness of the Cholesky factorization computed
 * Cholesky functions with the following criteria:
 *
 *    \f[ ||L'L-A||_oo/(||A||_oo.N.eps) < 60. \f]
 *
 *  or
 *
 *    \f[ ||UU'-A||_oo/(||A||_oo.N.eps) < 60. \f]
 *
 *  where A is the original matrix, and L, or U, the result of the Cholesky
 *  factorization.
 *
 *******************************************************************************
 *
 * @param[in,out] parsec
 *          The parsec context of the application that will run the operation.
 *
 * @param[in] loud
 *          The level of verbosity required.
 *
 * @param[in] uplo
 *          = dplasmaUpper: Upper triangle of A and A0 are referenced;
 *          = dplasmaLower: Lower triangle of A and A0 are referenced.
 *
 * @param[in] A
 *          Descriptor of the distributed matrix A result of the Cholesky
 *          factorization. Holds L or U. If uplo == dplasmaUpper, the only the
 *          upper part is referenced, otherwise if uplo == dplasmaLower, the
 *          lower part is referenced.
 *
 * @param[in] A0
 *          Descriptor of the original distributed matrix A before
 *          factorization. If uplo == dplasmaUpper, the only the upper part is
 *          referenced, otherwise if uplo == dplasmaLower, the lower part is
 *          referenced.
 *
 *******************************************************************************
 *
 * @return
 *          \retval 1, if the result is incorrect
 *          \retval 0, if the result is correct
 *
 ******************************************************************************/
int check_dpotrf( parsec_context_t *parsec, int loud,
                  dplasma_enum_t uplo,
                  parsec_tiled_matrix_dc_t *A,
                  parsec_tiled_matrix_dc_t *A0 )
{
    two_dim_block_cyclic_t *twodA = (two_dim_block_cyclic_t *)A0;
    two_dim_block_cyclic_t LLt;
    int info_factorization;
    double Rnorm = 0.0;
    double Anorm = 0.0;
    double result = 0.0;
    int M = A->m;
    int N = A->n;
    double eps = std::numeric_limits<double>::epsilon();
    dplasma_enum_t side;

    two_dim_block_cyclic_init(&LLt, matrix_RealDouble, matrix_Tile,
                              ttg::ttg_default_execution_context().size(), twodA->grid.rank,
                              A->mb, A->nb,
                              M, N,
                              0, 0,
                              M, N,
                              twodA->grid.krows, twodA->grid.kcols,
                              twodA->grid.rows /*twodA->grid.ip, twodA->grid.jq*/);

    LLt.mat = parsec_data_allocate((size_t)LLt.super.nb_local_tiles *
                                  (size_t)LLt.super.bsiz *
                                  (size_t)parsec_datadist_getsizeoftype(LLt.super.mtype));

    dplasma_dlaset( parsec, dplasmaUpperLower, 0., 0.,(parsec_tiled_matrix_dc_t *)&LLt );
    dplasma_dlacpy( parsec, uplo, A, (parsec_tiled_matrix_dc_t *)&LLt );

    /* Compute LL' or U'U  */
    side = (uplo == dplasmaUpper ) ? dplasmaLeft : dplasmaRight;
    dplasma_dtrmm( parsec, side, uplo, dplasmaTrans, dplasmaNonUnit, 1.0,
                   A, (parsec_tiled_matrix_dc_t*)&LLt);

    /* compute LL' - A or U'U - A */
    dplasma_dtradd( parsec, uplo, dplasmaNoTrans,
                    -1.0, A0, 1., (parsec_tiled_matrix_dc_t*)&LLt);

    Anorm = dplasma_dlansy(parsec, dplasmaInfNorm, uplo, A0);
    Rnorm = dplasma_dlansy(parsec, dplasmaInfNorm, uplo,
                           (parsec_tiled_matrix_dc_t*)&LLt);

    result = Rnorm / ( Anorm * N * eps ) ;

    if ( loud > 2 ) {
        printf("============\n");
        printf("Checking the Cholesky factorization \n");

        if ( loud > 3 )
            printf( "-- ||A||_oo = %e, ||L'L-A||_oo = %e\n", Anorm, Rnorm );

        printf("-- ||L'L-A||_oo/(||A||_oo.N.eps) = %e \n", result);
    }

    if ( std::isnan(Rnorm)  || std::isinf(Rnorm)  ||
         std::isnan(result) || std::isinf(result) ||
         (result > 60.0) )
    {
        if( loud ) printf("-- Factorization is suspicious ! \n");
        info_factorization = 1;
    }
    else
    {
        if( loud ) printf("-- Factorization is CORRECT ! \n");
        info_factorization = 0;
    }

    parsec_data_free(LLt.mat); LLt.mat = NULL;
    parsec_tiled_matrix_dc_destroy( (parsec_tiled_matrix_dc_t*)&LLt);

    return info_factorization;
}


/**
 *******************************************************************************
 *
 * @ingroup dplasma_double_check
 *
 * check_daxmb - Returns the result of the following test
 *
 *    \f[ (|| A x - b ||_oo / ((||A||_oo * ||x||_oo + ||b||_oo) * N * eps) ) < 60. \f]
 *
 *  where A is the original matrix, b the original right hand side, and x the
 *  solution computed through any factorization.
 *
 *******************************************************************************
 *
 * @param[in,out] parsec
 *          The parsec context of the application that will run the operation.
 *
 * @param[in] loud
 *          The level of verbosity required.
 *
 * @param[in] uplo
 *          = dplasmaUpper: Upper triangle of A is referenced;
 *          = dplasmaLower: Lower triangle of A is referenced.
 *
 * @param[in] A
 *          Descriptor of the distributed matrix A result of the Cholesky
 *          factorization. Holds L or U. If uplo == dplasmaUpper, the only the
 *          upper part is referenced, otherwise if uplo == dplasmaLower, the
 *          lower part is referenced.
 *
 * @param[in,out] b
 *          Descriptor of the original distributed right hand side b.
 *          On exit, b is overwritten by (b - A * x).
 *
 * @param[in] x
 *          Descriptor of the solution to the problem, x.
 *
 *******************************************************************************
 *
 * @return
 *          \retval 1, if the result is incorrect
 *          \retval 0, if the result is correct
 *
 ******************************************************************************/
int check_daxmb( parsec_context_t *parsec, int loud,
                 dplasma_enum_t uplo,
                 parsec_tiled_matrix_dc_t *A,
                 parsec_tiled_matrix_dc_t *b,
                 parsec_tiled_matrix_dc_t *x )
{
    int info_solution;
    double Rnorm = 0.0;
    double Anorm = 0.0;
    double Bnorm = 0.0;
    double Xnorm, result;
    int N = b->m;
    double eps = std::numeric_limits<double>::epsilon();

    Anorm = dplasma_dlansy(parsec, dplasmaInfNorm, uplo, A);
    Bnorm = dplasma_dlange(parsec, dplasmaInfNorm, b);
    Xnorm = dplasma_dlange(parsec, dplasmaInfNorm, x);

    /* Compute b - A*x */
    dplasma_dsymm( parsec, dplasmaLeft, uplo, -1.0, A, x, 1.0, b);

    Rnorm = dplasma_dlange(parsec, dplasmaInfNorm, b);

    result = Rnorm / ( ( Anorm * Xnorm + Bnorm ) * N * eps ) ;

    if ( loud > 2 ) {
        printf("============\n");
        printf("Checking the Residual of the solution \n");
        if ( loud > 3 )
            printf( "-- ||A||_oo = %e, ||X||_oo = %e, ||B||_oo= %e, ||A X - B||_oo = %e\n",
                    Anorm, Xnorm, Bnorm, Rnorm );

        printf("-- ||Ax-B||_oo/((||A||_oo||x||_oo+||B||_oo).N.eps) = %e \n", result);
    }

    if (std::isnan(Xnorm) || std::isinf(Xnorm) || std::isnan(result) || std::isinf(result) || (result > 60.0) ) {
        if( loud ) printf("-- Solution is suspicious ! \n");
        info_solution = 1;
    }
    else{
        if( loud ) printf("-- Solution is CORRECT ! \n");
        info_solution = 0;
    }

    return info_solution;
}

static void
dplasma_dprint_tile( int m, int n,
                     const parsec_tiled_matrix_dc_t* descA,
                     const double *M )
{
    int tempmm = ( m == descA->mt-1 ) ? descA->m - m*descA->mb : descA->mb;
    int tempnn = ( n == descA->nt-1 ) ? descA->n - n*descA->nb : descA->nb;
    int ldam = BLKLDD( descA, m );

    int ii, jj;

    fflush(stdout);
    for(ii=0; ii<tempmm; ii++) {
        if ( ii == 0 )
            fprintf(stdout, "(%2d, %2d) :", m, n);
        else
            fprintf(stdout, "          ");
        for(jj=0; jj<tempnn; jj++) {
#if defined(PRECISION_z) || defined(PRECISION_c)
            fprintf(stdout, " (% e, % e)",
                    creal( M[jj*ldam + ii] ),
                    cimag( M[jj*ldam + ii] ));
#else
            fprintf(stdout, " % e", M[jj*ldam + ii]);
#endif
        }
        fprintf(stdout, "\n");
    }
    fflush(stdout);
    usleep(1000);
}