// SPDX-License-Identifier: BSD-3-Clause
//
// Dense 2.5D SUMMA for block-structured GEMM using TTG.
// Uses the make_tt functional interface (no TT subclasses).
//
// Computes C = A * B where A (MT×KT tiles), B (KT×NT tiles), C (MT×NT tiles)
// are dense matrices partitioned into uniform tiles.

#include <algorithm>
#include <iostream>
#include <limits>
#include <random>
#include <sys/time.h>
#include <utility>
#include <vector>

#if __has_include(<btas/features.h>)
#  include <btas/features.h>
#  ifdef BTAS_IS_USABLE
#    include <btas/btas.h>
#    include <btas/util/mohndle.h>
#    include <btas/optimize/contract.h>
#  else
#    error "BTAS found but unusable (Boost.Iterators missing)"
#  endif
#else
#  error "BTAS not found; gemm.cc requires BTAS"
#endif

#include "ttg.h"
#include "ttg/util/multiindex.h"
#include "ttg/util/future.h"
#include "ttg/util/bug.h"

#include "devicetensor.h"
#include "devicegemm.h"

using namespace ttg;

// ---------------------------------------------------------------------------
// Device / host selection
// ---------------------------------------------------------------------------

#if defined(TTG_ENABLE_CUDA)
#  define HAVE_GEMM_DEVICE 1
static constexpr ttg::ExecutionSpace space = ttg::ExecutionSpace::CUDA;
#elif defined(TTG_ENABLE_HIP)
#  define HAVE_GEMM_DEVICE 1
static constexpr ttg::ExecutionSpace space = ttg::ExecutionSpace::HIP;
#elif defined(TTG_ENABLE_LEVEL_ZERO)
#  define HAVE_GEMM_DEVICE 1
static constexpr ttg::ExecutionSpace space = ttg::ExecutionSpace::L0;
#else
static constexpr ttg::ExecutionSpace space = ttg::ExecutionSpace::Host;
#endif

// ---------------------------------------------------------------------------
// Tile type
// ---------------------------------------------------------------------------

using scalar_t = double;

#if defined(HAVE_GEMM_DEVICE)
using blk_t = DeviceTensor<scalar_t,
                           btas::DEFAULT::range,
                           btas::mohndle<btas::varray<scalar_t,
                                                      ttg::pinned_allocator_t<scalar_t>>,
                                         btas::Handle::shared_ptr>>;
#else
using blk_t = btas::Tensor<scalar_t,
                           btas::DEFAULT::range,
                           btas::mohndle<btas::varray<scalar_t>,
                                         btas::Handle::shared_ptr>>;
#endif

// ---------------------------------------------------------------------------
// PaRSEC split-metadata descriptor
// ---------------------------------------------------------------------------

#if defined(TTG_USE_PARSEC)
namespace ttg {
  template <>
  struct SplitMetadataDescriptor<blk_t> {
    static auto get_metadata(const blk_t &b) {
      std::pair<int, int> dim{0, 0};
      if (!b.empty()) {
        assert(b.range().extent().size() == 2);
        std::get<0>(dim) = (int)b.range().extent(0);
        std::get<1>(dim) = (int)b.range().extent(1);
      }
      return dim;
    }
    static auto get_data(blk_t &b) {
      if (!b.empty())
        return boost::container::small_vector<iovec, 1>(
            1, iovec{b.size() * sizeof(scalar_t), b.data()});
      else
        return boost::container::small_vector<iovec, 1>{};
    }
    static auto create_from_metadata(const std::pair<int, int> &meta) {
      if (meta != std::pair{0, 0})
        return blk_t(btas::Range(std::get<0>(meta), std::get<1>(meta)));
      else
        return blk_t{};
    }
  };
}  // namespace ttg
#endif  // TTG_USE_PARSEC

// Boost serialization traits for blk_t (required for PaRSEC backend)
#include "ttg/serialization/backends/boost.h"
namespace ttg::detail {
  template <typename Archive>
  inline static constexpr bool is_boost_serializable_v<Archive, blk_t> = is_boost_archive_v<Archive>;
  template <typename Archive>
  inline static constexpr bool is_boost_serializable_v<Archive, const blk_t> = is_boost_archive_v<Archive>;
}  // namespace ttg::detail

// ---------------------------------------------------------------------------
// Host-side GEMM via BTAS contract (accumulated: C += A*B)
// ---------------------------------------------------------------------------

namespace btas {
  template <typename T_, class Range_, class Store_>
  void gemm(btas::Tensor<T_, Range_, Store_> &C,
            const btas::Tensor<T_, Range_, Store_> &A,
            const btas::Tensor<T_, Range_, Store_> &B) {
    using array = btas::DEFAULT::index<int>;
    if (C.empty()) {
      C = btas::Tensor<T_, Range_, Store_>(btas::Range(A.range().extent(0), B.range().extent(1)));
      btas::contract_222(1.0, A, array{1, 2}, B, array{2, 3}, 0.0, C, array{1, 3}, false, false);
    } else {
      btas::contract_222(1.0, A, array{1, 2}, B, array{2, 3}, 1.0, C, array{1, 3}, false, false);
    }
  }
}  // namespace btas

// ---------------------------------------------------------------------------
// Key type and distribution helpers
// ---------------------------------------------------------------------------

template <std::size_t Rank>
using Key = MultiIndex<Rank>;

/// Maps tile index {i,j} to MPI rank in a P×Q×R 3-D process grid.
inline int ij2rank(int i, int j, int P, int Q, int R) {
  const int p = i % P;
  const int q = j % Q;
  const int l = (i * j) % R;
  return (l * P * Q) + (q * P) + p;
}

/// Maps tile index {i,j,k} to MPI rank in a P×Q×R 3-D process grid.
inline int ijk2rank(int i, int j, int k, int P, int Q, int R) {
  const int p = i % P;
  const int q = j % Q;
  const int l = k % R;
  return (l * P * Q) + (q * P) + p;
}

// ---------------------------------------------------------------------------
// Norm helpers (for verification)
// ---------------------------------------------------------------------------

static std::tuple<double, double> norms(double t) {
  return {t * t, std::abs(t)};
}

template <typename T_, class Range_, class Store_>
static auto norms(const btas::Tensor<T_, Range_, Store_> &t) {
  using T = decltype(std::abs(std::declval<T_>()));
  T norm2sq = 0, norminf = 0;
  for (auto elem : t) {
    auto [sq, inf] = norms(elem);
    norm2sq += sq;
    norminf = std::max(norminf, inf);
  }
  return std::make_tuple(norm2sq, norminf);
}

// ---------------------------------------------------------------------------
// CLI helpers
// ---------------------------------------------------------------------------

static char *getCmdOption(char **begin, char **end, const std::string &option) {
  static char *empty = (char *)"";
  char **itr = std::find(begin, end, option);
  if (itr != end && ++itr != end) return *itr;
  return empty;
}

static bool cmdOptionExists(char **begin, char **end, const std::string &option) {
  return std::find(begin, end, option) != end;
}

static int cmdOptionIndex(char **begin, char **end, const std::string &option) {
  char **itr = std::find(begin, end, option);
  if (itr != end) return (int)(itr - begin);
  return -1;
}

static int parseOption(std::string &option, int default_value) {
  if (option.empty()) return default_value;
  size_t pos = option.find(':');
  if (pos == std::string::npos) pos = option.length();
  int N = std::stoi(option.substr(0, pos));
  option.erase(0, pos + 1);
  return N;
}

static long parseOption(std::string &option, long default_value) {
  if (option.empty()) return default_value;
  size_t pos = option.find(':');
  if (pos == std::string::npos) pos = option.length();
  long N = std::stol(option.substr(0, pos));
  option.erase(0, pos + 1);
  return N;
}

// ---------------------------------------------------------------------------
// Tile generation helper
// ---------------------------------------------------------------------------

static blk_t make_tile(int rows, int cols, unsigned long base_seed, int tile_i, int tile_j) {
  blk_t tile(btas::Range(rows, cols));
  std::mt19937_64 rng(base_seed ^ ((unsigned long)tile_i * 100003UL + tile_j));
  std::uniform_real_distribution<scalar_t> dist(-1.0, 1.0);
  for (auto &v : tile) v = dist(rng);
  return tile;
}

// ---------------------------------------------------------------------------
// make_read_dense
//
// Triggered once per process coordinate {p,q,r} by the Control task.
// Sends every locally-owned tile of the matrix to the broadcast flow.
// ---------------------------------------------------------------------------

template <typename Blk>
auto make_read_dense(const char *label, int rows, int cols,
                     const std::vector<Blk> &tiles,
                     ttg::Edge<Key<3>> &ctl,
                     ttg::Edge<Key<2>, Blk> &out,
                     std::function<int(const Key<3> &)> read_keymap,
                     std::function<int(const Key<2> &)> ij_keymap) {
  auto tt = ttg::make_tt(
      [rows, cols, &tiles, ij_keymap](const Key<3> & /*pqr*/,
                                      std::tuple<Out<Key<2>, Blk>> &out) {
        const int rank = ttg::default_execution_context().rank();
        for (int i = 0; i < rows; i++)
          for (int j = 0; j < cols; j++)
            if (ij_keymap(Key<2>{i, j}) == rank)
              ::send<0>(Key<2>{i, j}, ttg::persistent(tiles[i * cols + j]), out);
      },
      ttg::edges(ctl), ttg::edges(out),
      std::string("read_dense(") + label + ")",
      {"ctl"}, {std::string(label) + "_ij"});
  tt->set_keymap(std::move(read_keymap));
  return tt;
}

// ---------------------------------------------------------------------------
// make_write_dense
//
// Collects result tiles into a flat vector.
// Completion is guaranteed by fence() — no status future needed.
// ---------------------------------------------------------------------------

template <typename Blk>
auto make_write_dense(int cols, std::vector<Blk> &tiles,
                      ttg::Edge<Key<2>, Blk> &in,
                      std::function<int(const Key<2> &)> keymap) {
  auto tt = ttg::make_tt(
      [cols, &tiles](const Key<2> &key, Blk &&tile, std::tuple<> & /*out*/) {
        tiles[key[0] * cols + key[1]] = std::move(tile);
      },
      ttg::edges(in), ttg::edges(),
      "write_dense", {"Cij"}, {});
  tt->set_keymap(std::move(keymap));
  return tt;
}

// ---------------------------------------------------------------------------
// make_gemm_ttg
//
// Dense 2.5D SUMMA: C = A * B.
//
// Returns all internal TTs as a vector so the caller can keep them alive.
// Edges a_in / b_in / c_out connect to the surrounding read/write tasks.
// ---------------------------------------------------------------------------

template <ttg::ExecutionSpace Space = space, typename Blk = blk_t>
auto make_gemm_ttg(ttg::Edge<Key<2>, Blk> &a_in,
                   ttg::Edge<Key<2>, Blk> &b_in,
                   ttg::Edge<Key<2>, Blk> &c_out,
                   int MT, int NT, int KT,
                   const std::vector<int> &mTiles,
                   const std::vector<int> &nTiles,
                   const std::vector<int> &kTiles,
                   std::function<int(const Key<2> &)> ij_keymap,
                   std::function<int(const Key<3> &)> ijk_keymap,
                   long P, long Q, long /*R*/,
                   bool enable_device_map = true) {

  // Internal edges
  ttg::Edge<Key<3>, Blk> local_a_ijk("local_a_ijk"), a_ijk("a_ijk");
  ttg::Edge<Key<3>, Blk> local_b_ijk("local_b_ijk"), b_ijk("b_ijk");
  ttg::Edge<Key<3>, Blk> c_ijk("c_ijk");
  ttg::Edge<Key<2>, Blk> c_ij_p("c_ij_p");

  // -----------------------------------------------------------------
  // BcastA: A[i][k] → one copy per destination process {i,k,p}
  // -----------------------------------------------------------------
  auto bcast_a = ttg::make_tt(
      [NT, ijk_keymap](const Key<2> &ik, Blk &&a_ik,
                       std::tuple<Out<Key<3>, Blk>> &out) {
        const int i = (int)ik[0], k = (int)ik[1];
        ttg::trace("BcastA(", i, ", ", k, ")");
        const auto world = default_execution_context();
        std::vector<bool> procmap(world.size(), false);
        std::vector<Key<3>> ikp_keys;
        for (int j = 0; j < NT; j++) {
          const int p = ijk_keymap(Key<3>{i, j, k});
          if (!procmap[p]) {
            ikp_keys.emplace_back(Key<3>{i, k, p});
            procmap[p] = true;
          }
        }
        ::broadcast<0>(ikp_keys, std::move(a_ik), out);
      },
      ttg::edges(a_in), ttg::edges(local_a_ijk),
      "GeMM25D::bcast_a", {"a_ik"}, {"a_ikp"});
  bcast_a->set_keymap(ij_keymap);
  bcast_a->set_priomap([](const Key<2> &k) {
    return std::numeric_limits<int>::max() - (int)k[0];
  });

  // -----------------------------------------------------------------
  // LocalBcastA: A[i][k][p] → A[i][j][k] for all local multiply tasks
  // -----------------------------------------------------------------
  auto local_bcast_a = ttg::make_tt(
      [NT, ijk_keymap](const Key<3> &ikp, Blk &&a_ik,
                       std::tuple<Out<Key<3>, Blk>> &out) {
        const int i = (int)ikp[0], k = (int)ikp[1], p = (int)ikp[2];
        assert(p == default_execution_context().rank());
        ttg::trace("LocalBcastA(", i, ", ", k, ", ", p, ")");
        std::vector<Key<3>> ijk_keys;
        for (int j = 0; j < NT; j++)
          if (ijk_keymap(Key<3>{i, j, k}) == p)
            ijk_keys.emplace_back(Key<3>{i, j, k});
        ::broadcast<0>(ijk_keys, std::move(a_ik), out);
      },
      ttg::edges(local_a_ijk), ttg::edges(a_ijk),
      "GeMM25D::local_bcast_a", {"a_ikp"}, {"a_ijk"});
  local_bcast_a->set_keymap([](const Key<3> &ikp) { return (int)ikp[2]; });

  // -----------------------------------------------------------------
  // BcastB: B[k][j] → one copy per destination process {k,j,p}
  // -----------------------------------------------------------------
  auto bcast_b = ttg::make_tt(
      [MT, ijk_keymap](const Key<2> &kj, Blk &&b_kj,
                       std::tuple<Out<Key<3>, Blk>> &out) {
        const int k = (int)kj[0], j = (int)kj[1];
        ttg::trace("BcastB(", k, ", ", j, ")");
        const auto world = default_execution_context();
        std::vector<bool> procmap(world.size(), false);
        std::vector<Key<3>> kjp_keys;
        for (int i = 0; i < MT; i++) {
          const int p = ijk_keymap(Key<3>{i, j, k});
          if (!procmap[p]) {
            kjp_keys.emplace_back(Key<3>{k, j, p});
            procmap[p] = true;
          }
        }
        ::broadcast<0>(kjp_keys, std::move(b_kj), out);
      },
      ttg::edges(b_in), ttg::edges(local_b_ijk),
      "GeMM25D::bcast_b", {"b_kj"}, {"b_kjp"});
  bcast_b->set_keymap(ij_keymap);
  bcast_b->set_priomap([](const Key<2> &k) {
    return std::numeric_limits<int>::max() - (int)k[1];
  });

  // -----------------------------------------------------------------
  // LocalBcastB: B[k][j][p] → B[i][j][k] for all local multiply tasks
  // -----------------------------------------------------------------
  auto local_bcast_b = ttg::make_tt(
      [MT, ijk_keymap](const Key<3> &kjp, Blk &&b_kj,
                       std::tuple<Out<Key<3>, Blk>> &out) {
        const int k = (int)kjp[0], j = (int)kjp[1], p = (int)kjp[2];
        assert(p == default_execution_context().rank());
        ttg::trace("LocalBcastB(", k, ", ", j, ", ", p, ")");
        std::vector<Key<3>> ijk_keys;
        for (int i = 0; i < MT; i++)
          if (ijk_keymap(Key<3>{i, j, k}) == p)
            ijk_keys.emplace_back(Key<3>{i, j, k});
        ::broadcast<0>(ijk_keys, std::move(b_kj), out);
      },
      ttg::edges(local_b_ijk), ttg::edges(b_ijk),
      "GeMM25D::local_bcast_b", {"b_kjp"}, {"b_ijk"});
  local_bcast_b->set_keymap([](const Key<3> &kjp) { return (int)kjp[2]; });

  // -----------------------------------------------------------------
  // MultiplyAdd: C[i][j] += A[i][k] * B[k][j], chaining along k
  //
  // Three inputs: A[i][k], B[k][j] (read-only), C running sum (moved).
  // Two outputs: the finished C[i][j] partial sum → ReduceC (output 0),
  //              or the updated running sum → next k step (output 1).
  // -----------------------------------------------------------------

#if defined(HAVE_GEMM_DEVICE)
  auto f_multiply = [MT, NT, KT, mTiles, nTiles, kTiles, ijk_keymap]
      (const Key<3> &ijk,
       const Blk &A, const Blk &B, Blk &&C,
       std::tuple<Out<Key<2>, Blk>, Out<Key<3>, Blk>> &result) -> ttg::device::Task {

    const int i = (int)ijk[0], j = (int)ijk[1], k = (int)ijk[2];
    const int my_rank = default_execution_context().rank();

    int next_k = -1;
    for (int nk = k + 1; nk < KT; nk++)
      if (ijk_keymap(Key<3>{i, j, nk}) == my_rank) { next_k = nk; break; }
    const bool have_next_k = (next_k >= 0);

    if (C.empty()) C = Blk(btas::Range(mTiles[i], nTiles[j]), scalar_t(0));

    co_await ttg::device::select(A.b, B.b, C.b);
    device_gemm(C, A, B);
    if (have_next_k)
      co_await ttg::device::forward(
          ttg::device::send<1>(Key<3>{i, j, next_k}, std::move(C), result));
    else
      co_await ttg::device::forward(
          ttg::device::send<0>(Key<2>{i, j}, std::move(C), result));
  };
  auto multiplyadd = ttg::make_tt<Space>(
      f_multiply,
      ttg::edges(a_ijk, b_ijk, c_ijk), ttg::edges(c_ij_p, c_ijk),
      "GeMM25D::MultiplyAdd", {"a_ijk", "b_ijk", "c_ijk"}, {"c_ij", "c_ijk"});
  if (enable_device_map) {
    const int num_devices = ttg::device::num_devices();
    const int gp = std::max(1, (int)std::sqrt((double)num_devices));
    const int gq = num_devices / gp;
    multiplyadd->set_devicemap([P, Q, gp, gq](const Key<3> &ijk) {
      return (int)((((ijk[0] / P) % gp) * gq) + ((ijk[1] / Q) % gq));
    });
  }
#else
  auto f_multiply = [MT, NT, KT, mTiles, nTiles, kTiles, ijk_keymap]
      (const Key<3> &ijk,
       const Blk &A, const Blk &B, Blk &&C,
       std::tuple<Out<Key<2>, Blk>, Out<Key<3>, Blk>> &result) {

    const int i = (int)ijk[0], j = (int)ijk[1], k = (int)ijk[2];
    const int my_rank = default_execution_context().rank();

    int next_k = -1;
    for (int nk = k + 1; nk < KT; nk++)
      if (ijk_keymap(Key<3>{i, j, nk}) == my_rank) { next_k = nk; break; }
    const bool have_next_k = (next_k >= 0);

    if (C.empty()) C = Blk(btas::Range(mTiles[i], nTiles[j]), scalar_t(0));

    btas::gemm(C, A, B);
    if (have_next_k)
      ::send<1>(Key<3>{i, j, next_k}, std::move(C), result);
    else
      ::send<0>(Key<2>{i, j}, std::move(C), result);
  };
  auto multiplyadd = ttg::make_tt(
      f_multiply,
      ttg::edges(a_ijk, b_ijk, c_ijk), ttg::edges(c_ij_p, c_ijk),
      "GeMM25D::MultiplyAdd", {"a_ijk", "b_ijk", "c_ijk"}, {"c_ij", "c_ijk"});
#endif

  multiplyadd->set_keymap(ijk_keymap);
  multiplyadd->set_priomap([KT, ijk_keymap](const Key<3> &ijk) {
    const int my_rank = ijk_keymap(ijk);
    const int i = (int)ijk[0], j = (int)ijk[1], k = (int)ijk[2];
    int32_t len = 0;
    for (int nk = k + 1; nk < KT; nk++)
      if (ijk_keymap(Key<3>{i, j, nk}) == my_rank) ++len;
    return len;
  });

  // Seed the c_ijk self-edge: inject a zero tile at the first k step on this
  // rank for every (i,j) pair that this rank will work on.
  {
    const int my_rank = ttg::default_execution_context().rank();
    for (int i = 0; i < MT; i++)
      for (int j = 0; j < NT; j++)
        for (int k = 0; k < KT; k++)
          if (ijk_keymap(Key<3>{i, j, k}) == my_rank) {
            Blk zero(btas::Range(mTiles[i], nTiles[j]), scalar_t(0));
            multiplyadd->template in<2>()->send(Key<3>{i, j, k}, std::move(zero));
            break;  // only the first k on this rank
          }
  }

  // -----------------------------------------------------------------
  // ReduceC: accumulate per-layer partial sums for C[i][j]
  // -----------------------------------------------------------------
  auto reduce_c = ttg::make_tt(
      [](const Key<2> &ij, Blk &&c_partial,
         std::tuple<Out<Key<2>, Blk>> &out) {
        ttg::trace("ReduceC(", ij[0], ", ", ij[1], ")");
        ::send<0>(ij, std::move(c_partial), out);
      },
      ttg::edges(c_ij_p), ttg::edges(c_out),
      "GeMM25D::reduce_c", {"c_ij(p)"}, {"c_ij"});
  reduce_c->set_keymap(ij_keymap);
  reduce_c->template set_input_reducer<0>([](Blk &acc, const Blk &contrib) {
    assert(acc.size() == contrib.size());
    for (std::size_t e = 0; e < acc.size(); e++)
      *(acc.data() + e) += *(contrib.data() + e);
  });

  // Set stream sizes: one contribution per distinct rank that handles any
  // (i,j,k) for a given (i,j).
  {
    const auto world   = ttg::default_execution_context();
    const int my_rank  = world.rank();
    const int world_sz = world.size();
    for (int i = 0; i < MT; i++)
      for (int j = 0; j < NT; j++)
        if (ij_keymap(Key<2>{i, j}) == my_rank) {
          std::vector<bool> mask(world_sz, false);
          for (int k = 0; k < KT; k++)
            mask[ijk_keymap(Key<3>{i, j, k})] = true;
          int n = (int)std::count(mask.begin(), mask.end(), true);
          if (n > 1)
            reduce_c->template set_argstream_size<0>(Key<2>{i, j}, n);
        }
  }

  // Return all TTs so the caller keeps them alive until fence().
  std::vector<std::unique_ptr<ttg::TTBase>> ops;
  ops.push_back(std::move(bcast_a));
  ops.push_back(std::move(local_bcast_a));
  ops.push_back(std::move(bcast_b));
  ops.push_back(std::move(local_bcast_b));
  ops.push_back(std::move(multiplyadd));
  ops.push_back(std::move(reduce_c));
  return ops;
}

// ---------------------------------------------------------------------------
// timed_measurement
// ---------------------------------------------------------------------------

static void timed_measurement(
    const std::vector<blk_t> &A_tiles,
    const std::vector<blk_t> &B_tiles,
    int MT, int NT, int KT,
    const std::vector<int> &mTiles,
    const std::vector<int> &nTiles,
    const std::vector<int> &kTiles,
    const std::function<int(const Key<2> &)> &ij_keymap,
    const std::function<int(const Key<3> &)> &ijk_keymap,
    double gflops, int ts, int P, int Q, int R,
    bool enable_device_map) {

  std::vector<blk_t> C_tiles(MT * NT);

  auto read_keymap = [&](const Key<3> &key) {
    return ijk2rank((int)key[0], (int)key[1], (int)key[2], P, Q, R);
  };

  // Control task via shared state
  struct PQR { int P, Q, R; };
  auto pqr = std::make_shared<PQR>(PQR{P, Q, R});

  ttg::Edge<Key<3>> ctl("control");
  ttg::Edge<Key<2>, blk_t> eA, eB, eC;

  auto control = ttg::make_tt(
      [pqr](std::tuple<Out<Key<3>>> &out) {
        for (int p = 0; p < pqr->P; p++)
          for (int q = 0; q < pqr->Q; q++)
            for (int r = 0; r < pqr->R; r++)
              ::sendk<0>(Key<3>{p, q, r}, out);
      },
      ttg::edges(), ttg::edges(ctl), "Control", {}, {"ctl"});

  auto a = make_read_dense<blk_t>("A", MT, KT, A_tiles, ctl, eA, read_keymap, ij_keymap);
  auto b = make_read_dense<blk_t>("B", KT, NT, B_tiles, ctl, eB, read_keymap, ij_keymap);
  auto c = make_write_dense<blk_t>(NT, C_tiles, eC, ij_keymap);
  auto gemm_ops = make_gemm_ttg(eA, eB, eC, MT, NT, KT, mTiles, nTiles, kTiles,
                                  ij_keymap, ijk_keymap, P, Q, R, enable_device_map);
  TTGUNUSED(a); TTGUNUSED(b); TTGUNUSED(c); TTGUNUSED(gemm_ops);

  auto connected = ttg::make_graph_executable(control.get());
  assert(connected);
  TTGUNUSED(connected);

  struct timeval start{0}, end{0}, diff{0};
  gettimeofday(&start, nullptr);
  if (ttg::default_execution_context().rank() == 0) control->invoke();
  fence();
  gettimeofday(&end, nullptr);
  timersub(&end, &start, &diff);
  const double tc = (double)diff.tv_sec + (double)diff.tv_usec / 1e6;

#if defined(TTG_USE_MADNESS)
  const std::string rt("MAD");
#elif defined(TTG_USE_PARSEC)
  const std::string rt("PARSEC");
#else
  const std::string rt("Unknown");
#endif
  if (ttg::default_execution_context().rank() == 0) {
    std::cout << "TTG-" << rt
              << " PxQxR= " << P << " " << Q << " " << R
              << " ndevices= " << ttg::device::num_devices()
              << " MT= " << MT << " NT= " << NT << " KT= " << KT
              << " ts= " << ts
              << " gflops= " << gflops
              << " seconds= " << tc
              << " gflops/s= " << gflops / tc
              << std::endl;
  }
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main(int argc, char **argv) {
  int cores = -1;
  std::string nbCoreStr(getCmdOption(argv, argv + argc, "-c"));
  cores = parseOption(nbCoreStr, cores);

  if (int dashdash = cmdOptionIndex(argv, argv + argc, "--") > -1) {
    initialize(argc - dashdash, argv + dashdash, cores);
  } else {
    initialize(1, argv, cores);
  }

  std::string debugStr(getCmdOption(argv, argv + argc, "-d"));
  auto debug = (unsigned int)parseOption(debugStr, 0);
  if (debug & (1 << 1)) {
    using ttg::Debugger;
    auto debugger = std::make_shared<Debugger>();
    Debugger::set_default_debugger(debugger);
    debugger->set_exec(argv[0]);
    debugger->set_prefix(ttg::default_execution_context().rank());
    debugger->set_cmd("gdb_xterm");
  }
  if (debug & (1 << 0)) {
    ttg::trace_on();
    TTBase::set_trace_all(true);
  }

  const int mpi_size = ttg::default_execution_context().size();
  const int mpi_rank = ttg::default_execution_context().rank();

  // Auto-select a roughly-cubic P×Q×R factorisation of mpi_size.
  int P = 1, Q = mpi_size, R = 1;
  {
    int best = mpi_size;
    for (int c = 1; c <= (int)std::cbrt((double)mpi_size); c++) {
      for (int p = 1; p <= (int)std::sqrt((double)mpi_size / c); p++) {
        if ((mpi_size % (p * c)) == 0) {
          int q = mpi_size / (p * c);
          if (std::abs(c - p - q) <= best) {
            best = std::abs(c - p - q);
            P = p; Q = q; R = c;
          }
        }
      }
    }
  }

  // Allow command-line overrides.
  {
    std::string PStr(getCmdOption(argv, argv + argc, "-P"));
    P = parseOption(PStr, P);
    std::string QStr(getCmdOption(argv, argv + argc, "-Q"));
    Q = parseOption(QStr, Q);
    std::string RStr(getCmdOption(argv, argv + argc, "-R"));
    R = parseOption(RStr, 1);
  }
  if (P * Q * R != mpi_size) {
    if (!cmdOptionExists(argv, argv + argc, "-Q") && (mpi_size % (P * R) == 0))
      Q = mpi_size / (P * R);
    else if (!cmdOptionExists(argv, argv + argc, "-P") && (mpi_size % (Q * R)) == 0)
      P = mpi_size / (Q * R);
    else if (!cmdOptionExists(argv, argv + argc, "-R") && (mpi_size % (Q * P)) == 0)
      R = mpi_size / (Q * P);
    else {
      if (mpi_rank == 0)
        std::cerr << P << "x" << Q << "x" << R
                  << " is not a valid process grid -- bailing out\n";
      MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
  }

  // Matrix dimensions and tile size.
  std::string Mstr(getCmdOption(argv, argv + argc, "-M"));
  int M = parseOption(Mstr, 256);
  std::string Nstr(getCmdOption(argv, argv + argc, "-N"));
  int N = parseOption(Nstr, 256);
  std::string Kstr(getCmdOption(argv, argv + argc, "-K"));
  int K = parseOption(Kstr, 256);
  std::string tsStr(getCmdOption(argv, argv + argc, "-t"));
  int ts = parseOption(tsStr, 64);

  const bool enable_device_map = !cmdOptionExists(argv, argv + argc, "--default-device-map");

  // Work-distribution strategy.
  enum class WORKDIST { A, B, C };
  WORKDIST dist = WORKDIST::C;
  if (cmdOptionExists(argv, argv + argc, "-D")) {
    std::string DStr(getCmdOption(argv, argv + argc, "-D"));
    if (DStr == "a") dist = WORKDIST::A;
    else if (DStr == "b") dist = WORKDIST::B;
  }

  // Tile counts (ceiling division).
  const int MT = (M + ts - 1) / ts;
  const int NT = (N + ts - 1) / ts;
  const int KT = (K + ts - 1) / ts;

  // Tile size vectors (last tile may be smaller).
  std::vector<int> mTiles(MT), nTiles(NT), kTiles(KT);
  for (int i = 0; i < MT; i++) mTiles[i] = std::min(ts, M - i * ts);
  for (int j = 0; j < NT; j++) nTiles[j] = std::min(ts, N - j * ts);
  for (int k = 0; k < KT; k++) kTiles[k] = std::min(ts, K - k * ts);

  // Keymaps.
  std::function<int(const Key<2> &)> ij_keymap = [P, Q, R](const Key<2> &ij) {
    return ij2rank((int)ij[0], (int)ij[1], P, Q, R);
  };
  std::function<int(const Key<3> &)> ijk_keymap;
  if (dist == WORKDIST::A) {
    ijk_keymap = [P, Q, R](const Key<3> &ijk) {
      return ij2rank((int)ijk[0], (int)ijk[2], P, Q, R);
    };
  } else if (dist == WORKDIST::B) {
    ijk_keymap = [P, Q, R](const Key<3> &ijk) {
      return ij2rank((int)ijk[2], (int)ijk[1], P, Q, R);
    };
  } else {
    ijk_keymap = [P, Q, R](const Key<3> &ijk) {
      return ij2rank((int)ijk[0], (int)ijk[1], P, Q, R);
    };
  }

  // Reproducible seed (broadcast from rank 0).
  std::string seedStr(getCmdOption(argv, argv + argc, "-s"));
  long seedL = parseOption(seedStr, 0L);
  unsigned long seed = (seedL == 0) ? (unsigned long)std::random_device{}() : (unsigned long)seedL;
  ttg_broadcast(ttg::default_execution_context(), seed, 0);
  if (mpi_rank == 0) std::cerr << "#seed=" << seed << std::endl;

  // Check vs. timing mode.
  std::string checkStr(getCmdOption(argv, argv + argc, "-x"));
  int check = parseOption(checkStr, !(argc >= 2));
  const bool timing = (check == 0);

  // GFlops.
  const double gflops = 2.0 * (double)M * (double)N * (double)K / 1e9;

  // In check mode every rank allocates all tiles (for the local reference on
  // rank 0). In timing mode only locally-owned tiles are allocated.
  const bool alloc_all = (check != 0);

  std::vector<blk_t> A_tiles(MT * KT);
  std::vector<blk_t> B_tiles(KT * NT);

  for (int i = 0; i < MT; i++)
    for (int k = 0; k < KT; k++)
      if (alloc_all || ij_keymap(Key<2>{i, k}) == mpi_rank)
        A_tiles[i * KT + k] = make_tile(mTiles[i], kTiles[k], seed, i, k);

  for (int k = 0; k < KT; k++)
    for (int j = 0; j < NT; j++)
      if (alloc_all || ij_keymap(Key<2>{k, j}) == mpi_rank)
        B_tiles[k * NT + j] = make_tile(kTiles[k], nTiles[j], seed + 1, k, j);

  std::string nbrunStr(getCmdOption(argv, argv + argc, "-n"));
  int nb_runs = parseOption(nbrunStr, 1);

  if (timing) {
    execute();
    for (int nrun = 0; nrun < nb_runs; nrun++) {
#if defined(TTG_USE_PARSEC)
      parsec_devices_release_memory();
#endif
      timed_measurement(A_tiles, B_tiles, MT, NT, KT,
                        mTiles, nTiles, kTiles,
                        ij_keymap, ijk_keymap,
                        gflops, ts, P, Q, R, enable_device_map);
#if defined(TTG_USE_PARSEC)
      parsec_devices_reset_load(default_execution_context().impl().context());
#endif
    }
  } else {
    // -----------------------------------------------------------------------
    // Correctness check
    // -----------------------------------------------------------------------
    std::vector<blk_t> C_tiles(MT * NT);

    // Route all C tiles to rank 0 for verification.
    std::function<int(const Key<2> &)> keymap_rank0 = [](const Key<2> &) { return 0; };
    std::function<int(const Key<3> &)> read_keymap   = [&](const Key<3> &key) {
      return ijk2rank((int)key[0], (int)key[1], (int)key[2], P, Q, R);
    };

    struct PQR { int P, Q, R; };
    auto pqr = std::make_shared<PQR>(PQR{P, Q, R});

    ttg::Edge<Key<3>> ctl("control");
    ttg::Edge<Key<2>, blk_t> eA, eB, eC;

    auto control = ttg::make_tt(
        [pqr](std::tuple<Out<Key<3>>> &out) {
          for (int p = 0; p < pqr->P; p++)
            for (int q = 0; q < pqr->Q; q++)
              for (int r = 0; r < pqr->R; r++)
                ::sendk<0>(Key<3>{p, q, r}, out);
        },
        ttg::edges(), ttg::edges(ctl), "Control", {}, {"ctl"});

    auto a = make_read_dense<blk_t>("A", MT, KT, A_tiles, ctl, eA, read_keymap, ij_keymap);
    auto b = make_read_dense<blk_t>("B", KT, NT, B_tiles, ctl, eB, read_keymap, ij_keymap);
    auto c = make_write_dense<blk_t>(NT, C_tiles, eC, keymap_rank0);
    auto gemm_ops = make_gemm_ttg(eA, eB, eC, MT, NT, KT, mTiles, nTiles, kTiles,
                                    ij_keymap, ijk_keymap, P, Q, R, enable_device_map);
    TTGUNUSED(a); TTGUNUSED(b); TTGUNUSED(c); TTGUNUSED(gemm_ops);

    if (mpi_rank == 0)
      std::cout << Dot{true}(control.get()) << std::endl;

    auto connected = ttg::make_graph_executable(control.get());
    assert(connected);
    TTGUNUSED(connected);

    if (mpi_rank == 0) control->invoke();
    execute();
    fence();

    if (mpi_rank == 0) {
      // Reference: C_ref[i][j] = sum_k A[i][k] * B[k][j]
      std::vector<blk_t> Cref(MT * NT);
      for (int i = 0; i < MT; i++)
        for (int j = 0; j < NT; j++)
          for (int k = 0; k < KT; k++)
            btas::gemm(Cref[i * NT + j], A_tiles[i * KT + k], B_tiles[k * NT + j]);

      double norm2sq = 0, norminf = 0;
      for (int i = 0; i < MT; i++) {
        for (int j = 0; j < NT; j++) {
          const blk_t &ref = Cref[i * NT + j];
          const blk_t &got = C_tiles[i * NT + j];
          assert(ref.size() == got.size());
          blk_t diff(ref.range());
          for (std::size_t e = 0; e < ref.size(); e++)
            *(diff.data() + e) = *(ref.data() + e) - *(got.data() + e);
          auto [sq, inf] = norms(diff);
          norm2sq += sq;
          norminf = std::max(norminf, inf);
        }
      }
      std::cout << "||Cref - C||_2      = " << std::sqrt(norm2sq) << "\n";
      std::cout << "||Cref - C||_inf    = " << norminf << "\n";
      if (norminf > 1e-9) {
        std::cerr << "VERIFICATION FAILED\n";
        ttg_abort();
      } else {
        std::cout << "VERIFICATION PASSED\n";
      }
    }
  }

  ttg_finalize();
  return 0;
}
