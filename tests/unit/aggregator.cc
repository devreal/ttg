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

  SECTION("chunked-even") {
    ttg::Edge<int, void> I2O;
    ttg::Edge<int, int> O2S;
    const auto nranks = ttg::default_execution_context().size();

    //ttg::trace_on();

    constexpr std::size_t K = 10;  // number of keys
    constexpr std::size_t T = 8;  // total values expected per key
    constexpr std::size_t C = 2;  // chunk (batch) size; evenly divides T
    constexpr std::size_t N = K * T;

    std::vector<std::atomic<std::size_t>> received(K);
    std::vector<std::atomic<std::size_t>> fires(K);
    for (auto &x : received) x = 0;
    for (auto &x : fires) x = 0;

    auto op = ttg::make_tt(
        [&](const int &n, std::tuple<ttg::Out<int, int>> &outs) {
          int key = n / T;
          // std::cout << "op sending key=" << key << " value=" << n << std::endl;
          ttg::send<0>(key, n, outs);
        },
        ttg::edges(I2O), ttg::edges(O2S));

    auto sink_op = ttg::make_tt(
        [&](const int &key, const ttg::Aggregator<int> &agg, std::tuple<> &) {
          CHECK(agg.size() > 0);
          //std::cout << "sink_op processing key=" << key << " with " << agg.size() << " values" << std::endl;
          CHECK(agg.size() <= C);
          received[key] += agg.size();
          fires[key]++;
        },
        ttg::edges(ttg::make_aggregator(O2S, T, C)), ttg::edges());

    op->set_keymap([=](const auto &key) { return nranks - 1; });
    make_graph_executable(op);
    ttg::execute(ttg::default_execution_context());
    if (ttg::default_execution_context().rank() == 0) {
      for (std::size_t i = 0; i < N; ++i) {
        op->invoke(i);
      }
    }
    ttg::ttg_fence(ttg::default_execution_context());

    for (std::size_t k = 0; k < K; ++k) {
      CHECK(received[k] == T);
      CHECK(fires[k] == (T + C - 1) / C);
    }
  }

  SECTION("chunked-remainder") {
    ttg::Edge<int, void> I2O;
    ttg::Edge<int, int> O2S;
    const auto nranks = ttg::default_execution_context().size();

    constexpr std::size_t K = 5;   // number of keys
    constexpr std::size_t T = 10;  // total values expected per key
    constexpr std::size_t C = 4;   // chunk size; does not evenly divide T (last chunk is a remainder of 2)
    constexpr std::size_t N = K * T;

    std::vector<std::atomic<std::size_t>> received(K);
    std::vector<std::atomic<std::size_t>> fires(K);
    for (auto &x : received) x = 0;
    for (auto &x : fires) x = 0;

    auto op = ttg::make_tt(
        [&](const int &n, std::tuple<ttg::Out<int, int>> &outs) {
          int key = n / T;
          ttg::send<0>(key, n, outs);
        },
        ttg::edges(I2O), ttg::edges(O2S));

    auto sink_op = ttg::make_tt(
        [&](const int &key, const ttg::Aggregator<int> &agg, std::tuple<> &) {
          CHECK(agg.size() > 0);
          CHECK(agg.size() <= C);
          received[key] += agg.size();
          fires[key]++;
        },
        ttg::edges(ttg::make_aggregator(O2S, T, C)), ttg::edges());

    op->set_keymap([=](const auto &key) { return nranks - 1; });
    make_graph_executable(op);
    ttg::execute(ttg::default_execution_context());
    if (ttg::default_execution_context().rank() == 0) {
      for (std::size_t i = 0; i < N; ++i) {
        op->invoke(i);
      }
    }
    ttg::ttg_fence(ttg::default_execution_context());

    for (std::size_t k = 0; k < K; ++k) {
      CHECK(received[k] == T);
      CHECK(fires[k] == (T + C - 1) / C);  // ceil(10/4) == 3: two full chunks of 4, one remainder of 2
    }
  }

  SECTION("chunk-size-exceeds-target") {
    // chunk_size >= target degenerates to a single firing (the pre-chunking behavior)
    ttg::Edge<int, void> I2O;
    ttg::Edge<int, int> O2S;
    const auto nranks = ttg::default_execution_context().size();

    constexpr std::size_t K = 6;
    constexpr std::size_t T = 3;
    constexpr std::size_t C = 10;  // > T, so no chunking should occur
    constexpr std::size_t N = K * T;

    std::vector<std::atomic<std::size_t>> fires(K);
    for (auto &x : fires) x = 0;

    auto op = ttg::make_tt(
        [&](const int &n, std::tuple<ttg::Out<int, int>> &outs) {
          int key = n / T;
          ttg::send<0>(key, n, outs);
        },
        ttg::edges(I2O), ttg::edges(O2S));

    auto sink_op = ttg::make_tt(
        [&](const int &key, const ttg::Aggregator<int> &agg, std::tuple<> &) {
          CHECK(agg.size() == T);
          fires[key]++;
        },
        ttg::edges(ttg::make_aggregator(O2S, T, C)), ttg::edges());

    op->set_keymap([=](const auto &key) { return nranks - 1; });
    make_graph_executable(op);
    ttg::execute(ttg::default_execution_context());
    if (ttg::default_execution_context().rank() == 0) {
      for (std::size_t i = 0; i < N; ++i) {
        op->invoke(i);
      }
    }
    ttg::ttg_fence(ttg::default_execution_context());

    for (std::size_t k = 0; k < K; ++k) {
      CHECK(fires[k] == 1);
    }
  }

  SECTION("is-final") {
    // a chunked aggregator's is_final() must be false for every firing but the last, and true for
    // exactly the last one (whether or not it's a full chunk).
    ttg::Edge<int, void> I2O;
    ttg::Edge<int, int> O2S;
    const auto nranks = ttg::default_execution_context().size();

    constexpr std::size_t K = 5;   // number of keys
    constexpr std::size_t T = 10;  // total values expected per key
    constexpr std::size_t C = 4;   // chunk size; last chunk is a remainder of 2
    constexpr std::size_t N = K * T;

    std::vector<std::atomic<std::size_t>> fires(K);
    std::vector<std::atomic<std::size_t>> final_fires(K);
    for (auto &x : fires) x = 0;
    for (auto &x : final_fires) x = 0;

    auto op = ttg::make_tt(
        [&](const int &n, std::tuple<ttg::Out<int, int>> &outs) {
          int key = n / T;
          ttg::send<0>(key, n, outs);
        },
        ttg::edges(I2O), ttg::edges(O2S));

    auto sink_op = ttg::make_tt(
        [&](const int &key, const ttg::Aggregator<int> &agg, std::tuple<> &) {
          // chunk forks for the same key are created in order but, once created, are independently
          // scheduled tasks -- their *execution* order across worker threads is not guaranteed to match
          // creation order. So is_final() can't be checked against "this is the k-th firing we observed";
          // instead just confirm that exactly one firing per key is ever flagged final.
          ++fires[key];
          if (agg.is_final()) final_fires[key]++;
        },
        ttg::edges(ttg::make_aggregator(O2S, T, C)), ttg::edges());

    op->set_keymap([=](const auto &key) { return nranks - 1; });
    make_graph_executable(op);
    ttg::execute(ttg::default_execution_context());
    if (ttg::default_execution_context().rank() == 0) {
      for (std::size_t i = 0; i < N; ++i) {
        op->invoke(i);
      }
    }
    ttg::ttg_fence(ttg::default_execution_context());

    for (std::size_t k = 0; k < K; ++k) {
      CHECK(fires[k] == (T + C - 1) / C);
      CHECK(final_fires[k] == 1);  // exactly one final firing per key
    }
  }

  SECTION("chunked-combine-via-reducer") {
    // A chunked aggregator only supports a single (aggregating) input, so accumulating a running
    // result across chunks is done by composing two ordinary TTs instead of adding "other inputs"
    // support to the aggregator itself: this TT reduces each chunk to a partial sum and forwards it;
    // a second, plain reducer-based TT combines the partial sums into the final per-key total. Since
    // the number of chunks per key isn't known in advance here (the per-key target varies), the
    // producer uses is_final() to tell the combiner -- via ttg::set_size -- exactly when to stop
    // expecting more contributions for that key.
    ttg::Edge<int, void> I2O;
    ttg::Edge<int, int> O2S;
    ttg::Edge<int, long> Partial2Combiner;
    const auto nranks = ttg::default_execution_context().size();

    // key k expects (k+1) values, so the number of chunks per key varies
    constexpr std::size_t K = 8;
    constexpr std::size_t N = K * (K + 1) / 2;
    constexpr std::size_t C = 3;

    std::vector<std::atomic<long>> final_sum(K);
    for (auto &x : final_sum) x = -1;
    std::atomic<std::size_t> combined = 0;

    auto op = ttg::make_tt(
        [&](const int &n, std::tuple<ttg::Out<int, int>> &outs) {
          int key = 0;
          std::size_t base = 0;
          while (base + (key + 1) <= static_cast<std::size_t>(n)) {
            base += (key + 1);
            ++key;
          }
          ttg::send<0>(key, n, outs);
        },
        ttg::edges(I2O), ttg::edges(O2S));

    auto chunk_op = ttg::make_tt(
        [&](const int &key, const ttg::Aggregator<int> &agg, std::tuple<ttg::Out<int, long>> &outs) {
          long partial = 0;
          for (auto &&v : agg) partial += v;
          if (agg.is_final()) {
            // chunk forks for the same key execute independently and may complete out of order, so the
            // total chunk count must be known analytically (it's deterministic from key and C) rather
            // than counted at runtime -- is_final() only tells us *when* (on which firing) to declare it.
            std::size_t target = static_cast<std::size_t>(key) + 1;
            std::size_t n_chunks = (target + C - 1) / C;
            ttg::set_size<0>(key, n_chunks, outs);
          }
          ttg::send<0>(key, partial, outs);
        },
        ttg::edges(ttg::make_aggregator(O2S, [](const int &key) -> std::size_t { return key + 1; }, C)),
        ttg::edges(Partial2Combiner));

    auto combiner_op = ttg::make_tt(
        [&](const int &key, long &&sum, std::tuple<> &) {
          final_sum[key] = sum;
          combined++;
        },
        ttg::edges(Partial2Combiner), ttg::edges());
    combiner_op->set_input_reducer<0>([](long &a, const long &b) { a += b; });

    op->set_keymap([=](const auto &key) { return nranks - 1; });
    make_graph_executable(op);
    ttg::execute(ttg::default_execution_context());
    if (ttg::default_execution_context().rank() == 0) {
      for (std::size_t i = 0; i < N; ++i) {
        op->invoke(i);
      }
    }
    ttg::ttg_fence(ttg::default_execution_context());

    CHECK(combined == K / nranks);
    for (std::size_t k = 0; k < K; ++k) {
      long expected = 0;
      std::size_t base = 0;
      for (std::size_t kk = 0; kk < k; ++kk) base += (kk + 1);
      for (std::size_t n = base; n < base + (k + 1); ++n) expected += static_cast<long>(n);
      CHECK(final_sum[k] == expected);
    }
  }
}  // TEST_CASE("aggregator")
