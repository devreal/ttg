#include <algorithm>
#include <iostream>
#include <limits>
#include <random>
#include <sys/time.h>
#include <utility>
#include <vector>

#include "pmw.h"
#include "matrixtile.h"


#if defined(TTG_HAVE_CUDART)
#define ES ttg::ExecutionSpace::CUDA
#define TASKRET -> ttg::device::Task
#include <cusolverDn.h>
#elif defined(TTG_ENABLE_HIP)
#define ES ttg::ExecutionSpace::HIP
#define TASKRET -> ttg::device::Task
#include <hipsolver/hipsolver.h>
#include <hipblas/hipblas.h>
#else
#define ES ttg::ExecutionSpace::Host
#define TASKRET -> void
#endif



// ---------------------------------------------------------------------------
// CLI helpers
// ---------------------------------------------------------------------------

static char *getCmdOption(char **begin, char **end, const std::string &option);

static bool cmdOptionExists(char **begin, char **end, const std::string &option);

static int cmdOptionIndex(char **begin, char **end, const std::string &option);

static int parseOption(std::string &option, int default_value);

static long parseOption(std::string &option, long default_value);

template <std::size_t Rank>
using Key = MultiIndex<Rank>;

template<typename MatrixT>
auto make_gemm(MatrixT&& A, MatrixT& B, MatrixT& C,
               ttg::Edge<Key<2>, typename MatrixT::tile_type> in_a,
               ttg::Edge<Key<2>, typename MatrixT::tile_type> in_b) {
  using value_type = typename MatrixT::value;
  using tile_type = typename MatrixT::tile_type;
  ttg::Edge<Key<2>, typename MatrixT::tile_type> output;
  ttg::Edge<Key<2>, typename MatrixT::tile_type> i2g; // initial to symbolic
  ttg::Edge<Key<2>, typename MatrixT::tile_type> s2g; // symbolic to gemm
  ttg::Edge<Key<3>, typename MatrixT::tile_type> next_k;

  auto keymap = [=](const auto &ij) {
     return C.rank_of(ij[0], ij[1]);
   };

  /**
   * Sends an empty tile to the symbolic analysis TT.
   */
  auto initial_dispatcher_tt = ttg::make_tt(
    [=](const Key<2>& key, const tile_type& a) TASKRET {
      ttg::send<0>(Key<3>{key[0], key[1], 0}, tile_type());
    }, ttg::edges(in_a), ttg::edges(i2g), "InitialDispatcher");

  initial_dispatcher_tt->set_keymap(keymap);

  auto symbolic_analysis_tt = ttg::make_tt(
    [=](const Key<3>& key, const tile_type& a, const tile_type& b, const tile_type& c) TASKRET {

      // do symbolic analysis to determine the sparsity pattern of the output tile
      auto col_counts = ttg::Buffer<uint64_t, Allocator<uint64_t>>(b.cols());
      // used to return the total number of nonzeroes from the kernel
      auto nnz_buffer = ttg::Buffer<uint64_t, Allocator<uint64_t>>(1);

      // NOTE: for GPUs, we need to make the buffers available

      symbolic_analysis_kernel(a, b, c, col_counts, nnz_buffer);

      // allocate a new tile that will hold the result of the GEMM
      tile_type new_c = tile_type(c.cols(), nnz_buffer.host_ptr()[0]);

      // send the sparsity pattern to the next GEMM
      ttg::send<0>(key, std::move(new_c));
    }, ttg::edges(in_a, in_b, ttg::fuse(next_k, i2g)), ttg::edges(s2g));

  symbolic_analysis_tt->set_keymap(keymap);

  auto gemm_tt = ttg::make_tt(
    [=](const Key<3>& key, const tile_type& a, const tile_type& b, tile_type&& c, tile_type&& new_c) -> TASKRET {
      int K = key[2];
      /**
       * TODO: Tell TTG what data we need on the device and have it allocate space if needed.
       */
      //co_await ttg::device::select(a.row_indices(), a.col_indices(), a.values(),
      //                             b.row_indices(), b.col_indices(), b.values(),
      //                             c.row_indices(), c.col_indices(), c.values(),
      //                             new_c.row_indices(), new_c.col_indices(), new_c.values()
      //                            );

      sparse_gemm_kernel(a, b, c, new_c);

      if (K < A.cols()) {
        // recurse to next GEMM / symbolic analysis
        ttg::send<0>(Key<3>{key[0], key[1], K+1}, std::move(c), out);
      } else {
        // done, send to output
        ttg::send<1>(Key<2>{key[0], key[1]}, std::move(c), out);
      }

    }, ttg::edges(in_a, in_b, ttg::fuse(next_k, i2g), s2g), ttg::edges(next_k, output));

  gemm_tt->set_keymap(keymap);

  auto ins = std::make_tuple(initial_dispatcher_tt->template in<0>());
  auto outs = std::make_tuple(gemm_tt->template out<0>());
  std::vector<std::unique_ptr<ttg::TTBase>> ops(3);
  ops[0] = std::move(initial_dispatcher_tt);
  ops[1] = std::move(symbolic_analysis_tt);
  ops[2] = std::move(gemm_tt);

  return std::make_pair(make_ttg(std::move(ops), ins, outs, "GEMM TTG"), output);
}


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



  // Allow command-line overrides.
  {
    std::string PStr(getCmdOption(argv, argv + argc, "-P"));
    P = parseOption(PStr, P);
    std::string QStr(getCmdOption(argv, argv + argc, "-Q"));
    Q = parseOption(QStr, Q);
  }
  if (P * Q != mpi_size) {
    if (!cmdOptionExists(argv, argv + argc, "-Q") && (mpi_size % (P) == 0))
      Q = mpi_size / (P);
    else if (!cmdOptionExists(argv, argv + argc, "-P") && (mpi_size % (Q)) == 0)
      P = mpi_size / (Q);
    else {
      if (mpi_rank == 0)
        std::cerr << P << "x" << Q
                  << " is not a valid process grid -- bailing out\n";
      MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
  }

  // Matrix dimensions provided by the input data
  int M = 0, N = 0, K = 0;

  // Matrix tile size.
  std::string MBStr(getCmdOption(argv, argv + argc, "-MB"));
  int MB = parseOption(tsStr, 64);
  std::string NBStr(getCmdOption(argv, argv + argc, "-NB"));
  int NB = parseOption(tsStr, 64);
  std::string KBStr(getCmdOption(argv, argv + argc, "-KB"));
  int KB = parseOption(tsStr, 64);

  // Tile counts (ceiling division).
  const int MT = (M + ts - 1) / ts;
  const int NT = (N + ts - 1) / ts;
  const int KT = (K + ts - 1) / ts;

  SparseTileMatrix A(M, K, MB, NB, P, Q);

  // TODO: read in data and fill tiles
  SparseTileMatrix A(M, K, MB, NB, P, Q);
  SparseTileMatrix B(K, N, NB, KB, P, Q);
  SparseTileMatrix C(M, N, MB, NB, P, Q);

  // TODO: calculate GFlops based on the number of nonzeroes in the input matrices
  const double gflops = 2.0 * (double)M * (double)N * (double)K / 1e9;

  std::string nbrunStr(getCmdOption(argv, argv + argc, "-n"));
  int nb_runs = parseOption(nbrunStr, 1);

  ttg::Edge<Key<2>, typename SparseTileMatrix::tile_type> a_edge, b_edge;
  ttg::Edge<void, void> ctl;
  auto [load_tt_A, a_edge] = make_load_tt(A, ctl, "A");
  auto [load_tt_B, b_edge] = make_load_tt(B, ctl, "B");
  auto [gemm_ttg, c_edge] = make_gemm(A, B, C, a_edge, b_edge);
  auto store_tt_C = make_store_tt(C, c_edge, "C");

  auto connected = make_graph_executable(&ctl);

  if (timing) {
    execute();
    for (int nrun = 0; nrun < nb_runs; nrun++) {
#if defined(TTG_USE_PARSEC)
      parsec_devices_release_memory();
#endif
      load_tt_A->invoke();
      load_tt_B->invoke();
      ttg::
#if defined(TTG_USE_PARSEC)
      parsec_devices_reset_load(default_execution_context().impl().context());
#endif
    }
  }

  ttg_finalize();
  return 0;
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
