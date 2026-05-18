#ifndef HAVE_MATRIX_H
#define HAVE_MATRIX_H

#include "allocator.h"

#include <ttg.h>
#include <memory>
#include <vector>


template<typename ValueT, typename IndexT, typename AllocatorT = Allocator<ValueT>>
class SparseTile {
  ttg::Buffer<IndexT, AllocatorT> m_col_indices;
  ttg::Buffer<IndexT, AllocatorT> m_row_indices;
  ttg::Buffer<ValueT, AllocatorT> m_values;

public:
  SparseTile() = default;

  /**
   * Allocate a sparse tile with the given number of columns and nonzeros.
   * The caller is responsible for filling in the row indices, column indices, and values.
   */
  SparseTile(size_t num_cols, size_t nnz)
  : m_col_indices(num_cols)
  , m_row_indices(nnz)
  , m_values(nnz)
  { }

  /**
   * Construct a sparse tile from the given row indices, column indices, and values. The sizes of the vectors must be consistent.
   * The data is copied into the tile's buffers.
   */
  SparseTile(const std::vector<IndexT>& row_indices, const std::vector<IndexT>& col_indices, const std::vector<ValueT>& values)
  : m_col_indices(col_indices.size())
  , m_row_indices(row_indices.size())
  , m_values(values.size())
  {
    std::copy_n(row_indices.data(), row_indices.size(), m_row_indices.host_ptr());
    std::copy_n(col_indices.data(), col_indices.size(), m_col_indices.host_ptr());
    std::copy_n(values.data(), values.size(), m_values.host_ptr());
  }

  ttg::Buffer<IndexT, AllocatorT>& row_indices() { return m_row_indices; }
  ttg::Buffer<IndexT, AllocatorT>& col_indices() { return m_col_indices; }
  ttg::Buffer<ValueT, AllocatorT>& values() { return m_values; }

  const ttg::Buffer<IndexT, AllocatorT>& row_indices() const { return m_row_indices; }
  const ttg::Buffer<IndexT, AllocatorT>& col_indices() const { return m_col_indices; }
  const ttg::Buffer<ValueT, AllocatorT>& values() const { return m_values; }

  bool empty() const {
    return m_values.empty();
  }

  template<typename Archive>
  void serialize(Archive& ar, const unsigned int version) {
    serialize(ar);
  }

  template<typename Archive>
  void serialize(Archive& ar) {
    ar & m_row_indices & m_col_indices & m_values;
  }
};

#ifdef TTG_SERIALIZATION_SUPPORTS_MADNESS
static_assert(madness::is_serializable_v<madness::archive::BufferOutputArchive, SparseTile<float, uint64_t>>);
#endif  // TTG_SERIALIZATION_SUPPORTS_MADNESS


/**
 * A distributed sparse matrix class that uses 2D-cyclic distribution of tiles. Each tile is a SparseTile.
 * Supports shallow copying and moving, but not deep copying.
 */
template<typename ValueT, typename IndexT, typename AllocatorT = Allocator<ValueT>>
class SparseTileMatrix {
public:
  using tile_type = SparseTile<ValueT, IndexT, AllocatorT>;
  using value_type = ValueT;
  using index_type = IndexT;
  using allocator_type = AllocatorT;

private:
  std::shared_ptr<std::vector<tile_type>> m_tiles;
  size_t m_rows;
  size_t m_cols;
  size_t m_tile_rows;
  size_t m_tile_cols;
  size_t m_pr;  // process grid rows
  size_t m_pc;  // process grid cols

  size_t tile_index(size_t i, size_t j) const {
    return (i % m_tile_rows) * m_tile_cols + (j % m_tile_cols);
  }

  int tile_rank(size_t i, size_t j) const {
    return ((i % m_pr) * m_pc) + (j % m_pc);
  }

public:
  SparseTileMatrix(size_t rows, size_t cols, size_t tile_rows, size_t tile_cols, size_t pr, size_t pc)
  : m_rows(rows)
  , m_cols(cols)
  , m_tile_rows(tile_rows)
  , m_tile_cols(tile_cols)
  , m_pr(pr)
  , m_pc(pc)
  , m_tiles(std::make_shared<std::vector<tile_type>>(tile_rows * tile_cols))
  { }

  SparseTileMatrix(size_t rows, size_t cols, size_t tile_rows, size_t tile_cols)
  : SparseTileMatrix(rows, cols, tile_rows, tile_cols, 0, 0)
  {
    const int mpi_size = ttg::default_execution_context().size();
    const int mpi_rank = ttg::default_execution_context().rank();

    // Auto-select a roughly-quadratic P×Q factorisation of mpi_size.
    int P = 1, Q = mpi_size;
    {
      int best = mpi_size;
      for (int p = 1; p <= (int)std::sqrt((double)mpi_size); p++) {
        if ((mpi_size % p) == 0) {
          int q = mpi_size / p;
          if (std::abs(p - q) <= best) {
            best = std::abs(p - q);
            P = p; Q = q;
          }
        }
      }
    }
    if (P * Q != mpi_size) {
      throw std::runtime_error("Unable to auto-select process grid for given MPI size");
    }
    m_pr = P;
    m_pc = Q;
  }

  SparseTileMatrix(const SparseTileMatrix& other) = default;
  SparseTileMatrix(SparseTileMatrix&& other) = default;
  SparseTileMatrix& operator=(const SparseTileMatrix& other) = default;
  SparseTileMatrix& operator=(SparseTileMatrix&& other) = default;

  // Get tile at position (i, j) using 2D-cyclic distribution
  tile_type& tile(size_t i, size_t j) {
    size_t tile_idx = tile_index(i, j);
    return (*m_tiles)[tile_idx];
  }

  const tile_type& tile(size_t i, size_t j) const {
    size_t tile_idx = tile_index(i, j);
    return (*m_tiles)[tile_idx];
  }

  tile_type operator()(size_t i, size_t j) {
    return tile(i, j);
  }

  const tile_type operator()(size_t i, size_t j) const {
    return tile(i, j);
  }

  size_t num_tiles() const { return m_tiles->size(); }

  int rank_of(size_t i, size_t j) const {
    return tile_rank(i, j);
  }

  bool is_local(size_t i, size_t j) const {
    return ttg::default_execution_context().rank() == rank_of(i, j);
  }

  size_t rows() const { return m_rows; }
  size_t cols() const { return m_cols; }
  size_t tile_rows() const { return m_tile_rows; }
  size_t tile_cols() const { return m_tile_cols; }
  size_t proc_rows() const { return m_pr; }
  size_t proc_cols() const { return m_pc; }
};

template<typename ValueT, typename IndexT, typename AllocatorT = Allocator<ValueT>>
auto make_load_tt(SparseTileMatrix<ValueT, IndexT, AllocatorT>& A, ttg::Edge<void, void> ctl, std::string name) {
  using tile_type = typename SparseTileMatrix<ValueT, IndexT, AllocatorT>::tile_type;
  ttg::Edge<Key<2>, tile_type> toop;

  auto load_tt = ttg::make_tt(
    [=](){
      for (size_t i = 0; i < A.tile_rows(); i++) {
        for (size_t j = 0; j < A.tile_cols(); j++) {
          Key<2> key{i, j};
          ttg::trace("Loading tile (", i, ", ", j, ")");
          ttg::send<0>(key, A(i, j));
        }
      }
     }, ttg::edges(ctl), ttg::edges(toop), "LoadMatrix " + name, {}, {"To Op"});

  return std::make_pair(std::move(load_tt), toop);
}

template<typename ValueT, typename IndexT, typename AllocatorT = Allocator<ValueT>>
auto make_store_tt(SparseTileMatrix<ValueT, IndexT, AllocatorT>& A,
                  ttg::Edge<Key<2>, typename SparseTileMatrix<ValueT, IndexT, AllocatorT>::tile_type> fromop,
                  std::string name) {
  using tile_type = typename SparseTileMatrix<ValueT, IndexT, AllocatorT>::tile_type;
  auto store_tt = ttg::make_tt(
    [&](const Key<2>& key, tile_type&& tile){
      size_t i = key[0];
      size_t j = key[1];
      ttg::trace("Storing tile (", i, ", ", j, ")");
      A(i, j) = std::move(tile);
     }, ttg::edges(fromop), ttg::edges(), "StoreMatrix " + name, {"From Op"}, {});

  return store_tt;
}

#endif // HAVE_MATRIX_H