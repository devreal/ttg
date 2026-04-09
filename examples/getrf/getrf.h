#ifndef GETRF_H
#define GETRF_H

#include "ttg.h"
#include "potrf/pmw.h"


template <typename MatrixT>
auto make_getrf(MatrixT& A,
                ttg::Edge<Key1, MatrixTile<typename MatrixT::element_type>>& from_input,
                ttg::Edge<Key1, MatrixTile<typename MatrixT::element_type>>& from_gemm,
                ttg::Edge<Key2, MatrixTile<typename MatrixT::element_type>>& to_result,
                ttg::Edge<Key2, MatrixTile<typename MatrixT::element_type>>& to_trsm) {
  ttg::Edge<Key2, MatrixTile<typename MatrixT::element_type>> output("output");
  using T = typename MatrixT::element_type;
  assert(A.cols() == A.rows());

  auto f = [=](const Key1& key, MatrixTile<T>&& tile) TASKRET {
    const int K = key[0];


    /* from here we have a device selected */

    /* do the LU factorization */

    /* send the tile to the output */
    std::vector<Key2> trsm;
    for (int i = K+1; i < A.rows(); i++) {
      trsm.push_back(Key2{K, i});
      trsm.push_back(Key2{i, K});
    }

    ttg::broadcast<0, 1>(std::make_tuple(Key2(K, K), std::move(trsm)), std::move(tile));
  };

  return ttg::make_tt(std::move(f), // task function
                      ttg::edges(ttg::fuse(from_input, from_gemm)), // input edges
                      ttg::edges(to_result, to_trsm), // output edges
                      "LU");
}


template <typename MatrixT>
auto make_trsm(MatrixT& A,
                ttg::Edge<Key2, MatrixTile<typename MatrixT::element_type>>& from_getrf, // A
                ttg::Edge<Key2, MatrixTile<typename MatrixT::element_type>>& from_input, // B from memory
                ttg::Edge<Key2, MatrixTile<typename MatrixT::element_type>>& from_gemm,  // B from previous gemm
                ttg::Edge<Key2, MatrixTile<typename MatrixT::element_type>>& to_result, // to memory
                ttg::Edge<Key3, MatrixTile<typename MatrixT::element_type>>& to_gemmA,  // to gemm A input
                ttg::Edge<Key3, MatrixTile<typename MatrixT::element_type>>& to_gemmB   // to gemm B input
              ) {


  using T = typename MatrixT::element_type;
  auto f = [=](const Key2& key, const MatrixTile<T>& tile_A, MatrixTile<T>&& tile_B) {
    const int M = key[0];
    const int N = key[1];

    bool is_lower = M > N;

    std::vector<Key3> gemm;

    /* TODO: do the trsm */

    if (is_lower) {
      /* TODO: call kernel */
      for (int n = N+1; n < A.cols(); n++) {
        gemm.push_back(Key3{M, n, N});
      }
      ttg::broadcast<0, 1>(std::make_tuple(key, std::move(gemm)), std::move(tile_B));
    } else {
      /* TODO: call kernel */
      for (int m = M+1; m < A.rows(); m++) {
        gemm.push_back(Key3{m, N, M});
      }
      ttg::broadcast<0, 2>(std::make_tuple(key, std::move(gemm)), std::move(tile_B));
    }
  }

  return ttg::make_tt(std::move(f), // task function
                      ttg::edges(from_getrf, ttg::fuse(from_input, from_gemm)), // input edges
                      ttg::edges(to_result, to_gemmA, to_gemmB), // output edges
                      "TRSM");
}


template <typename MatrixT>
auto make_gemm(MatrixT& A,
               ttg::Edge<Key3, MatrixTile<typename MatrixT::element_type>>& from_trsmA, // A from trsm
               ttg::Edge<Key3, MatrixTile<typename MatrixT::element_type>>& from_trsmB, // B from trsm
               ttg::Edge<Key3, MatrixTile<typename MatrixT::element_type>>& from_input, // C from memory
               ttg::Edge<Key2, MatrixTile<typename MatrixT::element_type>>& to_getrf,   // A of LU
               ttg::Edge<Key2, MatrixTile<typename MatrixT::element_type>>& to_trsm,    // to trsm
               ) {
  using T = typename MatrixT::element_type;
  ttg::Edge<Key2, MatrixTile<typename MatrixT::element_type>> gemm_recurse;
  auto f = [=](const Key3& key, const MatrixTile<T>& tile_A, const MatrixTile<T>& tile_B, MatrixTile<T>&& tile_C) {
    const int M = key[0];
    const int N = key[1];
    const int K = key[2];

    /* TODO: call GEMM kernel */


    std::vector<Key2> lu;
    std::vector<Key2> trsm;
    std::vector<Key2> gemm;
    if (M == K+1 && N == K+1) {
      lu.push_back(Key2{M, N});
    } else if (M > K+1 && N == K+1) {
      trsm.push_back(Key2{M, N});
    } else if (M == K+1 && N > K+1) {
      trsm.push_back(Key2{M, N});
    } else if (M > K+1 && N > K+1) {
      gemm.push_back(Key2{M, N, K+1});
    }
    ttg::broadcast<0, 1, 2>(std::make_tuple(std::move(lu), std::move(trsm), std::move(gemm)), std::move(tile_C));
  };

  return ttg::make_tt(std::move(f), // task function
                      ttg::edges(from_trsmA, from_trsmB, ttg::fuse(from_input, gemm_recurse)), // input edges
                      ttg::edges(to_getrf, to_trsm, gemm_recurse), // output edges
                      "GEMM");
};


auto make_getrf_ttg(MatrixT& A,
                    ttg::Edge<Key2, MatrixTile<typename MatrixT::element_type>>& from_input,
                    ttg::Edge<Key2, MatrixTile<typename MatrixT::element_type>>& to_result,
                    bool enable_device_map) {


  auto keymap1 = [=](const Key1& key) { return A.rank_of(key[0], key[0]); };
  auto keymap2 = [=](const Key2& key) { return A.rank_of(key[0], key[1]); };
  auto keymap3 = [=](const Key3& key) { return A.rank_of(key[0], key[1]); };


  /**
   * Set a device map, 2d block-cyclic
   */
  int num_devices = ttg::device::num_devices();
  int gp = std::sqrt(num_devices);
  int gq = (num_devices > 0) ? (num_devices / gp) : 1;
  auto mapper = [&A, gp,gq,num_devices](int i){
                  auto device = (((i/A.P())%gp)*gq) + (i/A.Q())%gq;
                  return device;
                };

  auto devmap1 = [=](const Key1& key) { return mapper(key[0]); };
  auto devmap2 = [=](const Key2& key) { return mapper(key[0]); };
  auto devmap3 = [=](const Key3& key) { return mapper(key[0]); };


  ttg::Edge<Key1, MatrixTile<typename MatrixT::element_type>> dispatch_to_getrf("to_getrf");
  ttg::Edge<Key2, MatrixTile<typename MatrixT::element_type>> dispatch_to_trsm("to_trsm");
  ttg::Edge<Key3, MatrixTile<typename MatrixT::element_type>> dispatch_to_gemm("to_gemm");
  ttg::Edge<Key3, MatrixTile<typename MatrixT::element_type>> gemm_to_getrf("gemm_to_getrf");
  auto dispatcher = [=](const Key2& key, MatrixTile<typename MatrixT::element_type>& tile) {
    if (key[0] == 0 && key[1] == 0) {
      // diagonal goes to getrf
      ttg::send<0>(Key1{0}, tile);
    } else if (key[0] == 0 || key[1] == 0) {
      // lower goes to trsm
      ttg::send<1>(key, tile);
    } else {
      // rest goes to gemm
      ttg::send<2>(Key3{key[0], key[1], 0}, tile);
    }
  };
  auto dispatch_tt = ttg::make_tt(dispatcher, ttg::edges(from_input), ttg::edges(dispatch_to_getrf, dispatch_to_trsm, dispatch_to_gemm), "Dispatcher");
  dispatch_tt->set_keymap(keymap2);
  ttg::Edge<Key2, MatrixTile<typename MatrixT::element_type>> getrf_to_trsm("to_trsm");
  ttg::Edge<Key2, MatrixTile<typename MatrixT::element_type>> gemm_to_trsm("to_trsm");
  ttg::Edge<Key2, MatrixTile<typename MatrixT::element_type>> trsm_to_gemmA("to_gemmA");
  ttg::Edge<Key2, MatrixTile<typename MatrixT::element_type>> trsm_to_gemmB("to_gemmB");
  ttg::Edge<Key2, MatrixTile<typename MatrixT::element_type>> gemm_to_getrf("to_getrf");
  auto getrf_tt = make_getrf(A, dispatch_to_getrf, gemm_to_getrf, to_result, getrf_to_trsm);
  getrf_tt->set_keymap(keymap1);
  if (enable_device_map) {
    getrf_tt->set_devmap(devmap1);
  }
  auto trsm_tt = make_trsm(A, getrf_to_trsm, dispatch_to_trsm, gemm_to_trsm, to_result, trsm_to_gemmA, trsm_to_gemmB);
  trsm_tt->set_keymap(keymap2);
  if (enable_device_map) {
    trsm_tt->set_devmap(devmap2);
  }
  auto gemm_tt = make_gemm(A, trsm_to_gemmA, trsm_to_gemmB, dispatch_to_gemm, gemm_to_getrf, gemm_to_trsm);
  gemm_tt->set_keymap(keymap3);
  if (enable_device_map) {
    gemm_tt->set_devmap(devmap3);
  }

  std::vector<std::unique_ptr<ttg::TTBase>> ops(4);
  auto ins = std::make_tuple(tt_dispatch->template in<0>());
  auto outs = std::make_tuple(tt_getrf->template out<0>(), tt_trsm->template out<0>());
  ops[0] = std::move(dispatch_tt);
  ops[1] = std::move(getrf_tt);
  ops[2] = std::move(trsm_tt);
  ops[3] = std::move(gemm_tt);

  return make_ttg(std::move(ops), ins, outs, "GETRF TTG");
}

#endif // GETRF_H
