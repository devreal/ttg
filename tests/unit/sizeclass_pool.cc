// SPDX-License-Identifier: BSD-3-Clause
#include <catch2/catch_all.hpp>

#include "ttg/util/sizeclass_pool.h"

#include <atomic>
#include <cstdint>
#include <cstring>
#include <deque>
#include <mutex>
#include <random>
#include <set>
#include <thread>
#include <vector>

namespace {

  using ttg::detail::SizeClasses;
  using ttg::detail::SizeClassPool;
  using ttg::detail::SizeClassPoolConfig;

  struct UpstreamStats {
    std::atomic<std::size_t> allocs{0};
    std::atomic<std::size_t> deallocs{0};
    std::atomic<std::size_t> bytes_in_use{0};
  };

  /// thread-safe malloc-based upstream that counts calls
  struct CountingUpstream {
    static constexpr std::size_t alignment = 16;
    UpstreamStats* stats = nullptr;

    void* allocate(std::size_t bytes) {
      stats->allocs++;
      stats->bytes_in_use += bytes;
      return ::operator new(bytes, static_cast<std::align_val_t>(alignment));
    }
    void deallocate(void* p, std::size_t bytes) {
      stats->deallocs++;
      stats->bytes_in_use -= bytes;
      ::operator delete(p, static_cast<std::align_val_t>(alignment));
    }
  };

  template <typename Tag>
  using TestPool = SizeClassPool<CountingUpstream, Tag>;

  /// fills a block with a pattern derived from (owner, seq) and checks it later
  void stamp(void* p, std::size_t n, std::uint64_t tag) {
    auto* w = static_cast<unsigned char*>(p);
    std::memcpy(w, &tag, std::min(n, sizeof tag));
    if (n >= 2 * sizeof tag) std::memcpy(w + n - sizeof tag, &tag, sizeof tag);
  }
  bool check(const void* p, std::size_t n, std::uint64_t tag) {
    auto* r = static_cast<const unsigned char*>(p);
    std::uint64_t a = 0, b = 0;
    std::memcpy(&a, r, std::min(n, sizeof a));
    if (n < 2 * sizeof tag) return std::memcmp(&a, &tag, std::min(n, sizeof tag)) == 0;
    std::memcpy(&b, r + n - sizeof b, sizeof b);
    return a == tag && b == tag;
  }

}  // namespace

TEST_CASE("SizeClasses", "[sizeclass_pool]") {
  // every size maps to the smallest class that fits it, with at most 12.5% waste above 512 B
  for (std::size_t n = 1; n <= (std::size_t{1} << 22); n += (n < 4096 ? 1 : 997)) {
    const int c = SizeClasses::index(n);
    const std::size_t s = SizeClasses::size(c);
    REQUIRE(s >= n);
    REQUIRE(s % SizeClasses::kGranule == 0);
    if (c > 0) REQUIRE(SizeClasses::size(c - 1) < n);
    if (n > 512) REQUIRE(s - n <= n / 8);
  }
  for (int c = 0; c < SizeClasses::kMaxClasses; ++c) {
    REQUIRE(SizeClasses::index(SizeClasses::size(c)) == c);
    if (c > 0) REQUIRE(SizeClasses::size(c) > SizeClasses::size(c - 1));
  }
}

TEST_CASE("SizeClassPool basics", "[sizeclass_pool]") {
  struct Tag {};
  UpstreamStats stats;
  {
    TestPool<Tag> pool({}, CountingUpstream{&stats});

    SECTION("reuse and alignment") {
      void* a = pool.allocate(100);
      REQUIRE(reinterpret_cast<std::uintptr_t>(a) % TestPool<Tag>::alignment == 0);
      REQUIRE(reinterpret_cast<std::uintptr_t>(a) % SizeClasses::kGranule == 0);
      pool.deallocate(a, 100);
      void* b = pool.allocate(110);  // same class (128 B): LIFO thread cache hands back the same block
      REQUIRE(a == b);
      pool.deallocate(b, 110);
      REQUIRE(stats.allocs == 1);  // a single slab served everything
    }

    SECTION("zero-size and null") {
      void* z = pool.allocate(0);
      REQUIRE(z != nullptr);
      pool.deallocate(z, 0);
      pool.deallocate(nullptr, 64);
    }

    SECTION("many live blocks are distinct") {
      std::vector<void*> ptrs;
      std::set<void*> uniq;
      for (int i = 0; i < 20000; ++i) {
        const std::size_t n = 1 + (i * 37) % 9000;
        void* p = pool.allocate(n);
        stamp(p, n, i);
        ptrs.push_back(p);
        uniq.insert(p);
      }
      REQUIRE(uniq.size() == ptrs.size());
      for (int i = 0; i < 20000; ++i) {
        const std::size_t n = 1 + (i * 37) % 9000;
        REQUIRE(check(ptrs[i], n, i));
        pool.deallocate(ptrs[i], n);
      }
    }

    SECTION("requests above max_class_size bypass the pool") {
      const std::size_t big = pool.max_class_size() + 1;
      const auto before = stats.allocs.load();
      void* p = pool.allocate(big);
      REQUIRE(stats.allocs == before + 1);
      REQUIRE(pool.large_bytes_in_use() == big);
      pool.deallocate(p, big);
      REQUIRE(pool.large_bytes_in_use() == 0);
      REQUIRE(stats.deallocs == 1);
    }
  }
  // destroying the pool returns all slabs
  REQUIRE(stats.allocs == stats.deallocs);
  REQUIRE(stats.bytes_in_use == 0);
}

TEST_CASE("SizeClassPool uncached large classes", "[sizeclass_pool]") {
  struct Tag {};
  UpstreamStats stats;
  SizeClassPoolConfig cfg;
  cfg.max_slab_bytes = 1 << 20;
  cfg.thread_cache_class_bytes = 1 << 20;  // classes > 1 MiB are not cached per thread
  {
    TestPool<Tag> pool(cfg, CountingUpstream{&stats});
    const std::size_t n = 3 << 20;
    void* a = pool.allocate(n);
    REQUIRE(stats.allocs == 1);  // one block per slab for classes larger than a slab
    pool.deallocate(a, n);       // goes straight to the central list
    void* b = pool.allocate(n);
    REQUIRE(a == b);
    REQUIRE(stats.allocs == 1);  // recycled, no new upstream call
    pool.deallocate(b, n);
  }
  REQUIRE(stats.bytes_in_use == 0);
}

TEST_CASE("SizeClassPool returns surplus large blocks upstream", "[sizeclass_pool]") {
  struct Tag {};
  UpstreamStats stats;
  SizeClassPoolConfig cfg;
  cfg.max_slab_bytes = 1 << 20;
  cfg.thread_cache_class_bytes = 1 << 20;  // 3 MiB blocks are not cached per thread
  cfg.central_keep_bytes = 6 << 20;        // keep 2 free 3 MiB blocks
  {
    TestPool<Tag> pool(cfg, CountingUpstream{&stats});
    const std::size_t n = 3 << 20;
    std::vector<void*> ps;
    for (int i = 0; i < 10; ++i) ps.push_back(pool.allocate(n));
    REQUIRE(stats.allocs == 10);
    REQUIRE(pool.solo_count() == 10);
    for (void* p : ps) pool.deallocate(p, n);
    REQUIRE(pool.solo_count() == 2);
    REQUIRE(stats.bytes_in_use == 2 * n);
    // the kept blocks are reused without upstream calls
    void* a = pool.allocate(n);
    void* b = pool.allocate(n);
    REQUIRE(stats.allocs == 10);
    pool.deallocate(a, n);
    pool.deallocate(b, n);
  }
  REQUIRE(stats.allocs == stats.deallocs);
  REQUIRE(stats.bytes_in_use == 0);
}

TEST_CASE("SizeClassPool thread cache budget", "[sizeclass_pool]") {
  struct Tag {};
  UpstreamStats stats;
  SizeClassPoolConfig cfg;
  cfg.thread_cache_bytes = 1 << 20;
  cfg.thread_cache_class_bytes = 256 << 10;
  {
    TestPool<Tag> pool(cfg, CountingUpstream{&stats});
    std::vector<std::pair<void*, std::size_t>> blocks;
    for (int i = 0; i < 4000; ++i) {
      const std::size_t n = 64 + (i * 7919) % (128 << 10);
      blocks.emplace_back(pool.allocate(n), n);
    }
    std::size_t peak = 0;
    for (auto& [p, n] : blocks) {
      pool.deallocate(p, n);
      peak = std::max(peak, pool.thread_cached_bytes());
    }
    INFO("peak cached bytes: " << peak);
    REQUIRE(peak <= cfg.thread_cache_bytes + cfg.thread_cache_class_bytes);
    pool.flush_thread_cache();
    REQUIRE(pool.thread_cached_bytes() == 0);
  }
  REQUIRE(stats.bytes_in_use == 0);
}

TEST_CASE("SizeClassPool multithreaded with cross-thread frees", "[sizeclass_pool]") {
  struct Tag {};
  UpstreamStats stats;
  SizeClassPoolConfig cfg;
  cfg.min_slab_bytes = 16 << 10;   // small slabs to exercise growth races
  cfg.max_slab_bytes = 256 << 10;
  cfg.thread_cache_bytes = 1 << 20;  // small total budget to exercise budget flushes
  cfg.max_cached_blocks = 8;   // small caches to exercise flush/refill
  {
    TestPool<Tag> pool(cfg, CountingUpstream{&stats});

    const int nthreads = std::max(4u, std::min(16u, std::thread::hardware_concurrency()));
    constexpr int kIters = 20000;
    constexpr std::size_t kWindow = 64;

    struct Item {
      void* p;
      std::size_t n;
      std::uint64_t tag;
    };
    std::mutex handoff_mtx;
    std::deque<Item> handoff;  // blocks allocated by one thread and freed by another
    std::atomic<int> errors{0};

    auto worker = [&](int tid) {
      std::mt19937_64 rng(tid);
      std::uniform_int_distribution<int> kind(0, 9);
      std::vector<Item> live;
      for (int i = 0; i < kIters; ++i) {
        const int k = kind(rng);
        const std::size_t n = k < 6 ? 1 + rng() % 1024 : (k < 9 ? 1 + rng() % (64 << 10) : 1 + rng() % (2 << 20));
        const std::uint64_t tag = (std::uint64_t(tid) << 32) | std::uint64_t(i);
        void* p = pool.allocate(n);
        stamp(p, n, tag);
        live.push_back({p, n, tag});
        if (live.size() > kWindow) {
          const std::size_t victim = rng() % live.size();
          Item it = live[victim];
          live[victim] = live.back();
          live.pop_back();
          if (rng() % 2) {  // hand it to some other thread
            std::lock_guard lock(handoff_mtx);
            handoff.push_back(it);
          } else {
            if (!check(it.p, it.n, it.tag)) errors++;
            pool.deallocate(it.p, it.n);
          }
        }
        if (i % 4 == 0) {  // free something another thread allocated
          Item it{nullptr, 0, 0};
          {
            std::lock_guard lock(handoff_mtx);
            if (!handoff.empty()) {
              it = handoff.front();
              handoff.pop_front();
            }
          }
          if (it.p) {
            if (!check(it.p, it.n, it.tag)) errors++;
            pool.deallocate(it.p, it.n);
          }
        }
      }
      for (auto& it : live) {
        if (!check(it.p, it.n, it.tag)) errors++;
        pool.deallocate(it.p, it.n);
      }
      // thread exit flushes this thread's cache back to the pool
    };

    std::vector<std::thread> threads;
    for (int t = 0; t < nthreads; ++t) threads.emplace_back(worker, t);
    for (auto& t : threads) t.join();
    for (auto& it : handoff) {
      if (!check(it.p, it.n, it.tag)) errors++;
      pool.deallocate(it.p, it.n);
    }
    handoff.clear();
    REQUIRE(errors == 0);

    // a second round must be served from the recycled blocks without growing much
    const auto slabs_before = pool.slab_count();
    std::vector<std::thread> threads2;
    for (int t = 0; t < nthreads; ++t) threads2.emplace_back(worker, t + 1000);
    for (auto& t : threads2) t.join();
    for (auto& it : handoff) pool.deallocate(it.p, it.n);
    handoff.clear();
    REQUIRE(errors == 0);
    INFO("slabs before second round: " << slabs_before << ", after: " << pool.slab_count());
    CHECK(pool.slab_count() <= slabs_before + slabs_before / 4 + 4);
  }
  REQUIRE(stats.allocs == stats.deallocs);
  REQUIRE(stats.bytes_in_use == 0);
}

namespace {
  struct VecTag {};
  using VecPool = TestPool<VecTag>;
  UpstreamStats vec_stats;
  struct VecPoolGetter {
    VecPool& operator()() const {
      static VecPool* pool = new VecPool({}, CountingUpstream{&vec_stats});
      return *pool;
    }
  };
}  // namespace

TEST_CASE("sizeclass_allocator", "[sizeclass_pool]") {
  using A = ttg::detail::sizeclass_allocator<double, VecPoolGetter>;
  static_assert(std::allocator_traits<A>::is_always_equal::value);
  std::vector<double, A> v(1000, 1.0);
  v.resize(5000);  // default-init construct path
  for (int i = 0; i < 5000; ++i) v[i] = i;
  std::vector<double, A> w = v;
  REQUIRE(w[4999] == 4999.0);
  using B = std::allocator_traits<A>::rebind_alloc<int>;
  REQUIRE(A{} == B{});
  std::vector<int, B> u(10, 7);
  REQUIRE(u[9] == 7);
}
