// SPDX-License-Identifier: BSD-3-Clause
#include <catch2/catch_all.hpp>

#include "ttg.h"

TEST_CASE("aggregator", "[aggregator][core]") {
  SECTION("fixed-target") {
    ttg::Edge<int, void> I2O;
    ttg::Edge<int, int> O2S;
    const auto nranks = ttg::default_execution_context().size();

    constexpr std::size_t N = 40;
    constexpr std::size_t M = 4;  // values per key
    std::atomic<std::size_t> fired = 0;

    auto op = ttg::make_tt(
        [&](const int &n, std::tuple<ttg::Out<int, int>> &outs) {
          int key = n / M;
          ttg::send<0>(key, n, outs);
        },
        ttg::edges(I2O), ttg::edges(O2S));

    auto sink_op = ttg::make_tt(
        [&](const int &key, const ttg::Aggregator<int> &agg, std::tuple<> &) {
          CHECK(agg.size() == M);
          long sum = 0;
          for (auto &&v : agg) sum += v;
          long expected_sum = 0;
          for (std::size_t n = key * M; n < (key + 1) * M; ++n) expected_sum += static_cast<long>(n);
          CHECK(sum == expected_sum);
          fired++;
        },
        ttg::edges(ttg::make_aggregator(O2S, M)), ttg::edges());

    op->set_keymap([=](const auto &key) { return nranks - 1; });
    make_graph_executable(op);
    ttg::execute(ttg::default_execution_context());
    if (ttg::default_execution_context().rank() == 0) {
      for (std::size_t i = 0; i < N; ++i) {
        op->invoke(i);
      }
    }
    ttg::ttg_fence(ttg::default_execution_context());
    CHECK(fired == N / M / nranks);
  }

  SECTION("per-key-target") {
    ttg::Edge<int, void> I2O;
    ttg::Edge<int, int> O2S;
    const auto nranks = ttg::default_execution_context().size();

    // key k expects (k+1) values: 0,1,2,... -> targets 1,2,3,...
    // n values total for keys 0..K-1 is K*(K+1)/2; pick K=8 -> N=36
    constexpr std::size_t K = 8;
    constexpr std::size_t N = K * (K + 1) / 2;
    std::atomic<std::size_t> fired = 0;

    auto op = ttg::make_tt(
        [&](const int &n, std::tuple<ttg::Out<int, int>> &outs) {
          // assign n to key such that key k receives exactly (k+1) values
          int key = 0;
          std::size_t base = 0;
          while (base + (key + 1) <= static_cast<std::size_t>(n)) {
            base += (key + 1);
            ++key;
          }
          ttg::send<0>(key, n, outs);
        },
        ttg::edges(I2O), ttg::edges(O2S));

    auto sink_op = ttg::make_tt(
        [&](const int &key, const ttg::Aggregator<int> &agg, std::tuple<> &) {
          CHECK(agg.size() == static_cast<std::size_t>(key + 1));
          fired++;
        },
        ttg::edges(ttg::make_aggregator(O2S, [](const int &key) -> std::size_t { return key + 1; })), ttg::edges());

    op->set_keymap([=](const auto &key) { return nranks - 1; });
    make_graph_executable(op);
    ttg::execute(ttg::default_execution_context());
    if (ttg::default_execution_context().rank() == 0) {
      for (std::size_t i = 0; i < N; ++i) {
        op->invoke(i);
      }
    }
    ttg::ttg_fence(ttg::default_execution_context());
    CHECK(fired == K / nranks);
  }
}  // TEST_CASE("aggregator")
