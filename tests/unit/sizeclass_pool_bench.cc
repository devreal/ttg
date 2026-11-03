// SPDX-License-Identifier: BSD-3-Clause
//
// Microbenchmark: concurrent allocate/deallocate throughput and latency of the size-class pool
// versus single-lock pools. Not run by ctest.
//
// usage: sizeclass-pool-bench [threads=8] [ops_per_thread=200000] [window=256] [min_bytes=64]
//                             [max_bytes=4194304] [handoff_percent=25]
//
// Each thread keeps `window` live blocks with log-uniform random sizes in [min_bytes, max_bytes],
// replaces a random one per step, and hands `handoff_percent`% of its frees to the next thread
// (cross-thread frees, as when a task on another worker releases a tile). The live set is bounded
// by threads * (window + 64) blocks; "live" reports the steady-state value after the warm-up fill.

#include "ttg/config.h"
#include "ttg/util/sizeclass_pool.h"

#if defined(TTG_HAVE_DEVICE) && defined(__has_include)
#if __has_include(<umpire/Umpire.hpp>)
#define BENCH_HAVE_UMPIRE_PINNED 1
#include "ttg/env.h"
#endif
#endif

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <iostream>
#include <memory_resource>
#include <mutex>
#include <random>
#include <string>
#include <thread>
#include <vector>

namespace {

  struct Params {
    int threads = 8;
    long ops = 200000;
    std::size_t window = 256;
    std::size_t min_bytes = 64;
    std::size_t max_bytes = std::size_t{4} << 20;
    int handoff_percent = 25;
  };

  /// type-erased allocator under test
  struct Subject {
    std::string name;
    std::function<void*(std::size_t)> allocate;
    std::function<void(void*, std::size_t)> deallocate;
    std::function<std::size_t()> footprint = {};  ///< peak bytes held from the backing memory, if known
  };

  /// log2-bucketed latency histogram (ns)
  struct Histogram {
    std::array<std::uint64_t, 40> buckets{};
    std::uint64_t max_ns = 0;
    void add(std::uint64_t ns) {
      buckets[std::min<std::size_t>(39, ns ? 64 - __builtin_clzll(ns) : 0)]++;
      max_ns = std::max(max_ns, ns);
    }
    void merge(const Histogram& o) {
      for (std::size_t i = 0; i < buckets.size(); ++i) buckets[i] += o.buckets[i];
      max_ns = std::max(max_ns, o.max_ns);
    }
    /// upper bound (ns) of the bucket holding quantile q
    std::uint64_t quantile(double q) const {
      std::uint64_t total = 0;
      for (auto b : buckets) total += b;
      const auto target = static_cast<std::uint64_t>(std::ceil(q * total));
      std::uint64_t acc = 0;
      for (std::size_t i = 0; i < buckets.size(); ++i) {
        acc += buckets[i];
        if (acc >= target) return std::uint64_t{1} << i;
      }
      return max_ns;
    }
  };

  struct Block {
    void* p;
    std::size_t n;
  };

  /// mailbox for cross-thread frees; its lock is only shared by two neighboring threads
  constexpr std::size_t kMaxInbox = 64;

  struct alignas(64) Inbox {
    std::mutex mtx;
    std::vector<Block> blocks;
  };

  void run(const Subject& s, const Params& prm) {
    using clock = std::chrono::steady_clock;
    std::vector<Inbox> inboxes(prm.threads);
    std::vector<Histogram> alloc_h(prm.threads), free_h(prm.threads);
    std::atomic<int> ready{0};
    std::atomic<std::size_t> live_bytes{0};
    std::atomic<bool> go{false};

    auto timed_free = [&](Histogram& h, const Block& b) {
      const auto t0 = clock::now();
      s.deallocate(b.p, b.n);
      h.add(std::chrono::duration_cast<std::chrono::nanoseconds>(clock::now() - t0).count());
    };

    auto worker = [&](int tid) {
      std::mt19937_64 rng(1234 + tid);
      const double lmin = std::log2(double(prm.min_bytes)), lmax = std::log2(double(prm.max_bytes));
      std::uniform_real_distribution<double> lsize(lmin, lmax);
      std::uniform_int_distribution<int> pct(0, 99);
      auto& ah = alloc_h[tid];
      auto& fh = free_h[tid];
      auto& next_inbox = inboxes[(tid + 1) % prm.threads];
      auto& my_inbox = inboxes[tid];
      std::vector<Block> live, incoming;
      live.reserve(prm.window + 1);

      auto alloc_one = [&] {
        const auto n = static_cast<std::size_t>(std::exp2(lsize(rng)));
        const auto t0 = clock::now();
        void* p = s.allocate(n);
        ah.add(std::chrono::duration_cast<std::chrono::nanoseconds>(clock::now() - t0).count());
        static_cast<volatile char*>(p)[0] = 1;  // touch it
        live.push_back({p, n});
      };

      for (std::size_t i = 0; i < prm.window; ++i) alloc_one();  // warm-up fill (timed but unavoidable)
      std::size_t mine = 0;
      for (auto& b : live) mine += b.n;
      live_bytes += mine;
      ready++;
      while (!go.load(std::memory_order_acquire)) std::this_thread::yield();

      for (long op = 0; op < prm.ops; ++op) {
        alloc_one();
        const std::size_t v = rng() % live.size();
        Block b = live[v];
        live[v] = live.back();
        live.pop_back();
        bool handed_off = false;
        if (prm.threads > 1 && pct(rng) < prm.handoff_percent) {
          std::lock_guard lock(next_inbox.mtx);
          // bound the inbox so a descheduled neighbor cannot inflate the live set
          if (next_inbox.blocks.size() < kMaxInbox) {
            next_inbox.blocks.push_back(b);
            handed_off = true;
          }
        }
        if (!handed_off) timed_free(fh, b);
        if ((op & 63) == 0) {
          {
            std::lock_guard lock(my_inbox.mtx);
            incoming.swap(my_inbox.blocks);
          }
          for (auto& ib : incoming) timed_free(fh, ib);
          incoming.clear();
        }
      }
      for (auto& b : live) s.deallocate(b.p, b.n);
    };

    std::vector<std::thread> threads;
    for (int t = 0; t < prm.threads; ++t) threads.emplace_back(worker, t);
    while (ready.load() < prm.threads) std::this_thread::yield();
    for (auto& h : alloc_h) h = {};
    for (auto& h : free_h) h = {};
    const auto t0 = clock::now();
    go.store(true, std::memory_order_release);
    for (auto& t : threads) t.join();
    const double secs = std::chrono::duration<double>(clock::now() - t0).count();
    for (auto& ib : inboxes)
      for (auto& b : ib.blocks) s.deallocate(b.p, b.n);

    Histogram a, f;
    for (auto& h : alloc_h) a.merge(h);
    for (auto& h : free_h) f.merge(h);
    const double mops = 2.0 * prm.ops * prm.threads / secs / 1e6;
    const double mib = 1.0 / (1 << 20);
    const std::size_t fp = s.footprint ? s.footprint() : 0;
    std::printf("%-34s live ~%8.1f MiB peak footprint %9s | ", s.name.c_str(), live_bytes.load() * mib,
                fp ? (std::to_string(static_cast<long>(fp * mib)) + " MiB").c_str() : "n/a");
    std::printf("%8.2f Mops/s | alloc p50 %6llu ns p99.9 %8llu ns max %10.3f ms"
                " | free p50 %6llu ns p99.9 %8llu ns max %10.3f ms\n",
                mops, (unsigned long long)a.quantile(0.5), (unsigned long long)a.quantile(0.999),
                a.max_ns / 1e6, (unsigned long long)f.quantile(0.5), (unsigned long long)f.quantile(0.999),
                f.max_ns / 1e6);
  }

  // ---------- host-memory subjects (always available) ----------

  struct MallocUpstream {
    static constexpr std::size_t alignment = 16;
    void* allocate(std::size_t n) { return std::malloc(n); }
    void deallocate(void* p, std::size_t) { std::free(p); }
  };

  using HostPool = ttg::detail::SizeClassPool<MallocUpstream>;

}  // namespace

int main(int argc, char** argv) {
  Params prm;
  if (argc > 1) prm.threads = std::atoi(argv[1]);
  if (argc > 2) prm.ops = std::atol(argv[2]);
  if (argc > 3) prm.window = std::strtoull(argv[3], nullptr, 10);
  if (argc > 4) prm.min_bytes = std::strtoull(argv[4], nullptr, 10);
  if (argc > 5) prm.max_bytes = std::strtoull(argv[5], nullptr, 10);
  if (argc > 6) prm.handoff_percent = std::atoi(argv[6]);
  std::printf("threads=%d ops/thread=%ld window=%zu sizes=[%zu, %zu] B handoff=%d%%\n", prm.threads, prm.ops,
              prm.window, prm.min_bytes, prm.max_bytes, prm.handoff_percent);

  std::vector<Subject> subjects;

#ifdef BENCH_HAVE_UMPIRE_PINNED
  ttg::Env::initialize();
  // footprint: high-water mark of the pinned QuickPool. The sizeclass subject runs second and draws its
  // slabs from the same QuickPool, so its own footprint is the increase over the first run's value.
  subjects.push_back({"umpire QuickPool pinned + mutex",
                      [](std::size_t n) { return ttg::umpire_pinned_allocator_t<char>{}.allocate(n); },
                      [](void* p, std::size_t n) {
                        ttg::umpire_pinned_allocator_t<char>{}.deallocate(static_cast<char*>(p), n);
                      },
                      [] { return ttg::Env::instance()->pinned_allocator_getActualHighWatermark(); }});
  subjects.push_back({"sizeclass pinned",
                      [](std::size_t n) { return ttg::sizeclass_pinned_allocator_t<char>{}.allocate(n); },
                      [](void* p, std::size_t n) {
                        ttg::sizeclass_pinned_allocator_t<char>{}.deallocate(static_cast<char*>(p), n);
                      },
                      [] { return ttg::detail::pinned_pool().reserved_bytes_hwm(); }});
#endif

  // single global lock around a pooling resource (stand-in for the mutex-serialized pool)
  static std::mutex pmr_mtx;
  static std::pmr::unsynchronized_pool_resource pmr_pool(
      std::pmr::pool_options{0, std::max<std::size_t>(prm.max_bytes, 1)});
  subjects.push_back({"host pmr pool + global mutex",
                      [](std::size_t n) {
                        std::lock_guard lock(pmr_mtx);
                        return pmr_pool.allocate(n, 16);
                      },
                      [](void* p, std::size_t n) {
                        std::lock_guard lock(pmr_mtx);
                        pmr_pool.deallocate(p, n, 16);
                      }});
  subjects.push_back({"host malloc", [](std::size_t n) { return std::malloc(n); },
                      [](void* p, std::size_t) { std::free(p); }});
  static HostPool host_pool(ttg::detail::SizeClassPoolConfig::from_env("TTG_PINNED_POOL"));
  subjects.push_back({"sizeclass host (malloc upstream)", [](std::size_t n) { return host_pool.allocate(n); },
                      [](void* p, std::size_t n) { host_pool.deallocate(p, n); },
                      [] { return host_pool.reserved_bytes_hwm(); }});

  for (auto& s : subjects) run(s, prm);

#ifdef BENCH_HAVE_UMPIRE_PINNED
  ttg::detail::pinned_pool().print_stats(std::cout);
#endif
  host_pool.print_stats(std::cout);
  return 0;
}
