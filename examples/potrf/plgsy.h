#pragma once

#include <ttg.h>
#include "core_plgsy.h"
#include "norm.h"
#include "pmw.h"
#include "util.h"

//#define KOKKOS_PLGSY_HOST

template <typename T>
auto make_plgsy(MatrixT<T>& A, unsigned long bump, unsigned long random_seed, ttg::Edge<Key2, void>& input, ttg::Edge<Key2, MatrixTile<T>>& output) {

#if (defined(TTG_HAVE_CUDART) || defined(TTG_HAVE_HIP)) && defined(TTG_HAVE_KOKKOS)

  auto f = [=](const Key2& key, std::tuple< ttg::Out<Key2, MatrixTile<T>> >& out) TASKRET {

    const int I = key[0];
    const int J = key[1];
    if(ttg::tracing()) ttg::print("PLGSY( ", key, ") on rank ", A.rank_of(key[0], key[1]));
    assert(A.is_local(I, J));

    auto tile = A(I, J);
    int tempmm, tempnn, ldam;

    tempmm = (I==A.rows()-1) ? A.rows_in_matrix()-I*A.rows_in_tile() : A.rows_in_tile();
    tempnn = (J==A.cols()-1) ? A.cols_in_matrix()-J*A.cols_in_tile() : A.cols_in_tile();
    ldam   = A.rows_in_tile();

    // TODO: enable after debugging

#if defined(KOKKOS_PLGSY_HOST)
    co_await ttg::to_device();
    auto ptr = tile.data();
    CORE_plgsy_device((double)bump, tempmm, tempnn, ptr, ldam,
                      A.rows_in_matrix(), I*A.rows_in_tile(), J*A.cols_in_tile(), random_seed);
#else // KOKKOS_PLGSY_HOST
    co_await ttg::to_device(tile.buffer());
    auto ptr = tile.buffer().current_device_ptr();
    CORE_plgsy_device((double)bump, tempmm, tempnn, ptr, ldam,
                      A.rows_in_matrix(), I*A.rows_in_tile(), J*A.cols_in_tile(), random_seed);
#if defined(DEBUG_TILES_VALUES)
    T norm; // norm to be computed
    device_norm(tile, &norm);
#endif // DEBUG_TILES_VALUES
#endif // KOKKOS_PLGSY_HOST

    // TODO: DEBUG
    co_await ttg::wait_kernel(tile.buffer());

#if defined(DEBUG_TILES_VALUES) && !defined(KOKKOS_PLGSY_HOST)
    tile.set_norm(norm);
#endif // DEBUG_TILES_VALUES
    assert(tile.norm() != 0.0);
    std::cout << "PLGSY " << key << " buffer " << ptr << " norm " << tile.norm() << std::endl;

    co_await ttg::device::forward(ttg::device::send<0>(key, std::move(tile)));
  };

  return ttg::make_tt<ES>(f, ttg::edges(input), ttg::edges(output), "PLGSY", {"startup"}, {"output"});
#else // defined(TTG_HAVE_CUDART) || defined(TTG_HAVE_HIP)
  auto f = [=](const Key2& key, std::tuple< ttg::Out<Key2, MatrixTile<T>> >& out) {
    /* write back any tiles that are not in the matrix already */
    const int I = key[0];
    const int J = key[1];
    if(ttg::tracing()) ttg::print("PLGSY( ", key, ") on rank ", A.rank_of(key[0], key[1]));
    assert(A.is_local(I, J));

    auto tile = A(I, J);
    T *a = tile.data();
    int tempmm, tempnn, ldam;

    tempmm = (I==A.rows()-1) ? A.rows_in_matrix()-I*A.rows_in_tile() : A.rows_in_tile();
    tempnn = (J==A.cols()-1) ? A.cols_in_matrix()-J*A.cols_in_tile() : A.cols_in_tile();
    ldam   = A.rows_in_tile();

    CORE_plgsy((double)bump, tempmm, tempnn, a, ldam,
               A.rows_in_matrix(), I*A.rows_in_tile(), J*A.cols_in_tile(), random_seed);
#ifdef DEBUG_TILES_VALUES
    tile.set_norm(blas::nrm2(tile.size(), a, 1));
#endif // DEBUG_TILES_VALUES

    ttg::send<0>(key, std::move(tile), out);
  };

  return ttg::make_tt(f, ttg::edges(input), ttg::edges(output), "PLGSY", {"startup"}, {"output"});
#endif // defined(TTG_HAVE_CUDART) || defined(TTG_HAVE_HIP)
}

auto make_plgsy_ttg(MatrixT<double> &A, unsigned long bump, unsigned long random_seed, ttg::Edge<Key2, void>& startup, ttg::Edge<Key2, MatrixTile<double>>&result, bool defer_write) {
  auto keymap2 = [&](const Key2& key) {
    return A.rank_of(key[0], key[1]);
  };
  auto plgsy_tt = make_plgsy(A, bump, random_seed, startup, result);
  plgsy_tt->set_keymap(keymap2);
  plgsy_tt->set_defer_writer(defer_write);

  auto ins = std::make_tuple(plgsy_tt->template in<0>());
  auto outs = std::make_tuple(plgsy_tt->template out<0>());
  std::vector<std::unique_ptr<ttg::TTBase>> ops(1);
  ops[0] = std::move(plgsy_tt);

  return make_ttg(std::move(ops), ins, outs, "PLGSY TTG");
}
