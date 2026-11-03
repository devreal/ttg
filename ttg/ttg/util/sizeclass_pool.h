// SPDX-License-Identifier: BSD-3-Clause
//
// A concurrent size-class memory pool with per-thread caches.
//

#ifndef TTG_UTIL_SIZECLASS_POOL_H
#define TTG_UTIL_SIZECLASS_POOL_H

#include <algorithm>
#include <atomic>
#include <bit>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <memory>
#include <mutex>
#include <new>
#include <ostream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace ttg::detail {

  /// Tuning knobs of SizeClassPool. All sizes in bytes.
  struct SizeClassPoolConfig {
    /// requests larger than this bypass the pool and go straight to the upstream allocator
    std::size_t max_class_size = std::size_t{256} << 20;
    /// a slab of class size `s` is `clamp(s * slab_blocks, min_slab_bytes, max_slab_bytes)` bytes
    /// (but holds at least one block)
    std::size_t slab_blocks = 16;
    std::size_t min_slab_bytes = std::size_t{256} << 10;
    std::size_t max_slab_bytes = std::size_t{1} << 20;
    /// classes with one block per slab ("solo" classes, roughly >= max_slab_bytes) keep at most this
    /// many bytes (but at least one block) of free blocks centrally; surplus frees go back upstream
    std::size_t central_keep_bytes = std::size_t{32} << 20;
    /// per-thread, per-class cache budget; classes larger than this are not cached per thread
    std::size_t thread_cache_class_bytes = std::size_t{2} << 20;
    /// per-thread budget across all classes; when exceeded, all other classes' caches are returned
    std::size_t thread_cache_bytes = std::size_t{8} << 20;
    /// cap on the number of blocks a thread caches per class
    std::uint32_t max_cached_blocks = 64;

    /// Override defaults from environment variables `<prefix>_{MAX_CLASS_BYTES, SLAB_BLOCKS,
    /// MIN_SLAB_BYTES, MAX_SLAB_BYTES, CENTRAL_KEEP_BYTES, THREAD_CACHE_CLASS_BYTES, THREAD_CACHE_BYTES,
    /// MAX_CACHED_BLOCKS}`.
    /// Byte values accept K/M/G suffixes.
    static SizeClassPoolConfig from_env(const std::string& prefix) { return from_env(prefix, SizeClassPoolConfig{}); }
    static SizeClassPoolConfig from_env(const std::string& prefix, SizeClassPoolConfig cfg) {
      auto parse = [](const char* s, auto& out) {
        if (s == nullptr || *s == '\0') return;
        char* end = nullptr;
        unsigned long long v = std::strtoull(s, &end, 10);
        switch (end ? *end : '\0') {
          case 'k': case 'K': v <<= 10; break;
          case 'm': case 'M': v <<= 20; break;
          case 'g': case 'G': v <<= 30; break;
          default: break;
        }
        out = static_cast<std::remove_reference_t<decltype(out)>>(v);
      };
      parse(std::getenv((prefix + "_MAX_CLASS_BYTES").c_str()), cfg.max_class_size);
      parse(std::getenv((prefix + "_SLAB_BLOCKS").c_str()), cfg.slab_blocks);
      parse(std::getenv((prefix + "_MIN_SLAB_BYTES").c_str()), cfg.min_slab_bytes);
      parse(std::getenv((prefix + "_MAX_SLAB_BYTES").c_str()), cfg.max_slab_bytes);
      parse(std::getenv((prefix + "_CENTRAL_KEEP_BYTES").c_str()), cfg.central_keep_bytes);
      parse(std::getenv((prefix + "_THREAD_CACHE_CLASS_BYTES").c_str()), cfg.thread_cache_class_bytes);
      parse(std::getenv((prefix + "_THREAD_CACHE_BYTES").c_str()), cfg.thread_cache_bytes);
      parse(std::getenv((prefix + "_MAX_CACHED_BLOCKS").c_str()), cfg.max_cached_blocks);
      return cfg;
    }
  };

  /// Size-class arithmetic: 64-byte granules up to 512 bytes, then 8 classes per power of two
  /// (at most 12.5% internal fragmentation). Every class size is a multiple of 64.
  struct SizeClasses {
    static constexpr int kSubBits = 3;                         ///< log2(classes per power of two)
    static constexpr int kSub = 1 << kSubBits;                 ///< classes per power of two
    static constexpr int kGranuleBits = 6;                     ///< 64-byte granule
    static constexpr int kLinearBits = kGranuleBits + kSubBits;  ///< linear classes up to 2^9 = 512 bytes
    static constexpr int kMaxBits = 40;                        ///< largest supported class: 1 TiB
    static constexpr std::size_t kGranule = std::size_t{1} << kGranuleBits;
    static constexpr std::size_t kMaxSize = std::size_t{1} << kMaxBits;

    /// index of the smallest class that holds @p n bytes; requires 0 < n <= kMaxSize
    static constexpr int index(std::size_t n) noexcept {
      if (n <= (std::size_t{1} << kLinearBits)) {
        return n <= kGranule ? 0 : static_cast<int>((n + kGranule - 1) >> kGranuleBits) - 1;
      }
      const int k = 63 - std::countl_zero(static_cast<std::uint64_t>(n - 1));  // 2^k < n <= 2^(k+1)
      const int shift = k - kSubBits;
      const std::size_t sub = ((n - (std::size_t{1} << k)) + (std::size_t{1} << shift) - 1) >> shift;  // 1..kSub
      return kSub + (k - kLinearBits) * kSub + static_cast<int>(sub) - 1;
    }

    /// size in bytes of class @p c
    static constexpr std::size_t size(int c) noexcept {
      if (c < kSub) return static_cast<std::size_t>(c + 1) << kGranuleBits;
      const int k = kLinearBits + (c - kSub) / kSub;
      const std::size_t sub = static_cast<std::size_t>((c - kSub) % kSub + 1);
      return (std::size_t{1} << k) + (sub << (k - kSubBits));
    }

    /// number of classes up to kMaxSize (closed form of index(kMaxSize) + 1, checked below)
    static constexpr int kMaxClasses = kSub * (kMaxBits - kLinearBits + 1);
  };
  static_assert(SizeClasses::index(SizeClasses::kMaxSize) + 1 == SizeClasses::kMaxClasses);
  static_assert(SizeClasses::size(SizeClasses::kMaxClasses - 1) == SizeClasses::kMaxSize);

  /**
   * A thread-scalable pool that carves fixed-size blocks out of large slabs obtained from an
   * upstream allocator.
   *
   * - Each thread owns a cache of free blocks per size class: allocate/deallocate are a pop/push
   *   on a thread-private list (no atomics, no locks).
   * - Caches exchange blocks with a per-class central list in batches; the central critical
   *   section is O(1) (push/pop a whole batch, or bump a slab pointer).
   * - Upstream is called only to obtain a new slab (rare) or for requests above max_class_size.
   * - Each thread caches at most thread_cache_class_bytes per class and thread_cache_bytes in
   *   total, which bounds the memory idling in thread caches.
   * - No pointer lookup is ever needed: deallocate() must be given the same size as allocate(),
   *   which the C++ Allocator contract guarantees.
   * - Blocks freed on a different thread than they were allocated on simply join the freeing
   *   thread's cache; thread caches are returned to the central lists when the thread exits.
   *
   * Free blocks store the list links inside the block, so the memory must be host-accessible
   * (host or pinned host memory). Multi-block slabs are retained until the pool is destroyed;
   * blocks of solo classes (one block per upstream allocation) beyond central_keep_bytes per class
   * are returned to upstream when freed, which bounds the footprint of large blocks.
   *
   * @tparam Upstream provides `void* allocate(std::size_t)`, `void deallocate(void*, std::size_t)`
   *         and `static constexpr std::size_t alignment` (alignment of the pointers it returns);
   *         it is only called from rare paths and must be thread-safe itself
   * @tparam Tag distinguishes the per-thread caches of different pools; at most one live pool
   *         instance per (Upstream, Tag) pair may exist at a time
   *
   * @warning destroy a pool only after every other thread that used it has exited or called
   *          flush_thread_cache(), and after all memory it handed out has been returned
   */
  template <typename Upstream, typename Tag = Upstream>
  class SizeClassPool {
    struct FreeNode {
      FreeNode* next;        ///< next free block in the same list
      FreeNode* next_batch;  ///< (batch head only) next batch in the central stack
      std::size_t count;     ///< (batch head only) number of blocks in this batch
    };
    static_assert(sizeof(FreeNode) <= SizeClasses::kGranule);

    struct alignas(64) Central {
      std::mutex mtx;                 ///< protects batches/bump; held for O(1) work only
      FreeNode* batches = nullptr;    ///< stack of batches of free blocks
      std::size_t nfree = 0;          ///< number of blocks in batches
      char* bump = nullptr;           ///< untouched remainder of the newest slab
      char* bump_end = nullptr;
      std::mutex grow_mtx;            ///< serializes slab acquisition for this class
    };

    struct ThreadCache {
      struct Bin {
        FreeNode* head = nullptr;
        std::uint32_t count = 0;
      };
      SizeClassPool* owner = nullptr;
      std::size_t cached_bytes = 0;  ///< bytes held in all bins
      Bin bins[SizeClasses::kMaxClasses];

      ~ThreadCache() {
        if (owner) owner->drain(*this);
      }
    };

    static ThreadCache& thread_cache() noexcept {
      static thread_local ThreadCache tc;
      return tc;
    }

   public:
    /// alignment of every pointer returned by allocate()
    static constexpr std::size_t alignment = std::min<std::size_t>(SizeClasses::kGranule, Upstream::alignment);

    explicit SizeClassPool(SizeClassPoolConfig cfg = {}, Upstream upstream = {})
        : upstream_(std::move(upstream)), cfg_(sanitize(cfg)) {
      num_classes_ = SizeClasses::index(cfg_.max_class_size) + 1;
      max_class_size_ = SizeClasses::size(num_classes_ - 1);
      centrals_ = std::make_unique<Central[]>(num_classes_);
      class_size_.resize(num_classes_);
      slab_nblocks_.resize(num_classes_);
      keep_.resize(num_classes_);
      max_cached_.resize(num_classes_);
      batch_.resize(num_classes_);
      for (int c = 0; c < num_classes_; ++c) {
        const std::size_t s = SizeClasses::size(c);
        class_size_[c] = s;
        const std::size_t slab =
            std::clamp(s * cfg_.slab_blocks, cfg_.min_slab_bytes, cfg_.max_slab_bytes);
        slab_nblocks_[c] = std::max<std::size_t>(1, slab / s);
        keep_[c] = slab_nblocks_[c] == 1 ? std::max<std::size_t>(1, cfg_.central_keep_bytes / s)
                                         : std::numeric_limits<std::size_t>::max();
        const std::size_t by_bytes = std::min(cfg_.thread_cache_class_bytes, cfg_.thread_cache_bytes) / s;
        max_cached_[c] = static_cast<std::uint32_t>(std::min<std::size_t>(by_bytes, cfg_.max_cached_blocks));
        batch_[c] = std::max<std::uint32_t>(1, max_cached_[c] / 2);
      }
    }

    SizeClassPool(const SizeClassPool&) = delete;
    SizeClassPool& operator=(const SizeClassPool&) = delete;

    ~SizeClassPool() {
      flush_thread_cache();
      // solo blocks are individual upstream allocations; blocks of other classes live in slabs_
      for (int c = 0; c < num_classes_; ++c) {
        if (slab_nblocks_[c] != 1) continue;
        auto& central = centrals_[c];
        for (FreeNode* b = central.batches; b != nullptr;) {
          FreeNode* next_batch = b->next_batch;
          release_solo(c, b);
          b = next_batch;
        }
        if (central.bump != central.bump_end) {  // a grown block nobody picked up
          auto* b = reinterpret_cast<FreeNode*>(central.bump);
          b->next = nullptr;
          release_solo(c, b);
        }
      }
      for (auto& [ptr, bytes] : slabs_) upstream_.deallocate(ptr, bytes);
    }

    void* allocate(std::size_t bytes) {
      if (bytes == 0) bytes = 1;
      if (bytes > max_class_size_) return allocate_large(bytes);
      const int c = SizeClasses::index(bytes);
      auto& tc = bind(thread_cache());
      auto& bin = tc.bins[c];
      if (bin.head == nullptr) {
        refill(c, tc);
        if (tc.cached_bytes > cfg_.thread_cache_bytes) [[unlikely]] scavenge(tc, c);
      }
      FreeNode* n = bin.head;
      bin.head = n->next;
      --bin.count;
      tc.cached_bytes -= class_size_[c];
      return n;
    }

    /// @param bytes must equal the value passed to the allocate() call that returned @p ptr
    void deallocate(void* ptr, std::size_t bytes) noexcept {
      if (ptr == nullptr) return;
      if (bytes == 0) bytes = 1;
      if (bytes > max_class_size_) return deallocate_large(ptr, bytes);
      const int c = SizeClasses::index(bytes);
      auto& tc = bind(thread_cache());
      auto& bin = tc.bins[c];
      auto* n = static_cast<FreeNode*>(ptr);
      n->next = bin.head;
      bin.head = n;
      tc.cached_bytes += class_size_[c];
      if (++bin.count > max_cached_[c]) {
        flush(c, tc, batch_[c]);
      } else if (tc.cached_bytes > cfg_.thread_cache_bytes) [[unlikely]] {
        scavenge(tc, c);
      }
    }

    /// returns all blocks cached by the calling thread to the central lists
    void flush_thread_cache() noexcept {
      auto& tc = thread_cache();
      if (tc.owner == this) {
        drain(tc);
        tc.owner = nullptr;
      }
    }

    /// bytes currently held in the calling thread's cache
    std::size_t thread_cached_bytes() const noexcept {
      const auto& tc = thread_cache();
      return tc.owner == this ? tc.cached_bytes : 0;
    }

    /// bytes currently held from upstream by the size classes (in use, cached, or free)
    std::size_t reserved_bytes() const noexcept {
      return slab_bytes_.load(std::memory_order_relaxed) + solo_bytes_.load(std::memory_order_relaxed);
    }
    /// high-water mark of reserved_bytes()
    std::size_t reserved_bytes_hwm() const noexcept { return reserved_hwm_.load(std::memory_order_relaxed); }
    /// number of multi-block slabs obtained from upstream (retained until destruction)
    std::size_t slab_count() const noexcept { return slab_count_.load(std::memory_order_relaxed); }
    /// number of solo-class blocks currently held from upstream
    std::size_t solo_count() const noexcept { return solo_count_.load(std::memory_order_relaxed); }
    /// bytes currently allocated through the large (upstream pass-through) path
    std::size_t large_bytes_in_use() const noexcept { return large_bytes_.load(std::memory_order_relaxed); }
    /// largest request served from size classes; larger ones go to upstream
    std::size_t max_class_size() const noexcept { return max_class_size_; }
    const SizeClassPoolConfig& config() const noexcept { return cfg_; }
    Upstream& upstream() noexcept { return upstream_; }

    void print_stats(std::ostream& os) const {
      os << "SizeClassPool: " << reserved_bytes() << " bytes reserved (peak " << reserved_bytes_hwm() << "; "
         << slab_count() << " slabs, " << solo_count() << " solo blocks), " << large_bytes_in_use()
         << " large bytes in use; config: max class " << max_class_size_ << " B, slab " << cfg_.slab_blocks
         << " blocks in [" << cfg_.min_slab_bytes << ", " << cfg_.max_slab_bytes << "] B, central keep "
         << cfg_.central_keep_bytes << " B, thread cache " << cfg_.thread_cache_class_bytes << " B/class, "
         << cfg_.thread_cache_bytes << " B total, max " << cfg_.max_cached_blocks << " blocks/class\n";
    }

   private:
    static SizeClassPoolConfig sanitize(SizeClassPoolConfig cfg) {
      cfg.max_class_size = std::clamp<std::size_t>(cfg.max_class_size, SizeClasses::kGranule, SizeClasses::kMaxSize);
      cfg.slab_blocks = std::max<std::size_t>(cfg.slab_blocks, 1);
      cfg.max_slab_bytes = std::max<std::size_t>(cfg.max_slab_bytes, SizeClasses::kGranule);
      cfg.min_slab_bytes = std::min(cfg.min_slab_bytes, cfg.max_slab_bytes);
      return cfg;
    }

    ThreadCache& bind(ThreadCache& tc) noexcept {
      if (tc.owner != this) [[unlikely]] {
        // a different pool with the same Tag used this thread's cache: violates the class contract
        assert(tc.owner == nullptr && "at most one live SizeClassPool per (Upstream, Tag) is allowed");
        tc.owner = this;
      }
      return tc;
    }

    /// moves the first @p n blocks of the calling thread's bin @p c to the central list as one batch
    void flush(int c, ThreadCache& tc, std::uint32_t n) noexcept {
      auto& bin = tc.bins[c];
      assert(n >= 1 && n <= bin.count);
      tc.cached_bytes -= n * class_size_[c];
      FreeNode* head = bin.head;
      FreeNode* tail = head;
      for (std::uint32_t i = 1; i < n; ++i) tail = tail->next;
      bin.head = tail->next;
      bin.count -= n;
      tail->next = nullptr;
      push_batch(c, head, n);
    }

    void push_batch(int c, FreeNode* head, std::size_t n) noexcept {
      head->count = n;
      auto& central = centrals_[c];
      {
        std::lock_guard lock(central.mtx);
        if (central.nfree < keep_[c]) {  // always true for slab classes
          head->next_batch = central.batches;
          central.batches = head;
          central.nfree += n;
          return;
        }
      }
      release_solo(c, head);  // surplus large blocks go back upstream, outside the lock
    }

    /// returns every block of the null-terminated list @p head (solo class @p c) to upstream
    void release_solo(int c, FreeNode* head) noexcept {
      const std::size_t s = class_size_[c];
      std::size_t n = 0;
      for (FreeNode* b = head; b != nullptr;) {
        FreeNode* next = b->next;
        upstream_.deallocate(b, s);
        b = next;
        ++n;
      }
      solo_bytes_.fetch_sub(n * s, std::memory_order_relaxed);
      solo_count_.fetch_sub(n, std::memory_order_relaxed);
    }

    /// over the total budget: return all bins except @p keep_c (which is bounded by its own budget)
    void scavenge(ThreadCache& tc, int keep_c) noexcept {
      for (int c = 0; c < num_classes_; ++c) {
        auto& bin = tc.bins[c];
        if (c == keep_c || bin.count == 0) continue;
        tc.cached_bytes -= bin.count * class_size_[c];
        push_batch(c, bin.head, bin.count);
        bin = {};
      }
    }

    void drain(ThreadCache& tc) noexcept {
      for (int c = 0; c < num_classes_; ++c) {
        auto& bin = tc.bins[c];
        if (bin.count > 0) push_batch(c, bin.head, bin.count);  // the list is already null-terminated
        bin = {};
      }
      tc.cached_bytes = 0;
    }

    /// fills the calling thread's empty bin @p c with a batch of blocks
    void refill(int c, ThreadCache& tc) {
      auto& bin = tc.bins[c];
      const std::size_t s = class_size_[c];
      auto& central = centrals_[c];
      for (;;) {
        char* range = nullptr;
        std::size_t k = 0;
        {
          std::lock_guard lock(central.mtx);
          if (FreeNode* b = central.batches) {
            central.batches = b->next_batch;
            central.nfree -= b->count;
            bin.head = b;
            bin.count = static_cast<std::uint32_t>(b->count);
            tc.cached_bytes += b->count * s;
            return;
          }
          if (central.bump != central.bump_end) {
            k = std::min<std::size_t>(batch_[c], static_cast<std::size_t>(central.bump_end - central.bump) / s);
            range = central.bump;
            central.bump += k * s;
          }
        }
        if (range != nullptr) {
          // link the fresh blocks outside the lock
          for (std::size_t i = 0; i + 1 < k; ++i) {
            reinterpret_cast<FreeNode*>(range + i * s)->next = reinterpret_cast<FreeNode*>(range + (i + 1) * s);
          }
          reinterpret_cast<FreeNode*>(range + (k - 1) * s)->next = nullptr;
          bin.head = reinterpret_cast<FreeNode*>(range);
          bin.count = static_cast<std::uint32_t>(k);
          tc.cached_bytes += k * s;
          return;
        }
        grow(c);
      }
    }

    /// obtains a new slab for class @p c from upstream, unless another thread already did
    void grow(int c) {
      auto& central = centrals_[c];
      std::lock_guard grow_lock(central.grow_mtx);
      {
        std::lock_guard lock(central.mtx);
        if (central.batches != nullptr || central.bump != central.bump_end) return;
      }
      const std::size_t s = class_size_[c];
      const std::size_t nblocks = slab_nblocks_[c];
      if (nblocks == 1) {
        // solo block: exactly one upstream allocation, no padding (alignment is Upstream::alignment),
        // so the block pointer can be handed back to upstream as is
        void* raw = upstream_.allocate(s);
        if (raw == nullptr) throw std::bad_alloc{};
        solo_bytes_.fetch_add(s, std::memory_order_relaxed);
        solo_count_.fetch_add(1, std::memory_order_relaxed);
        update_hwm();
        std::lock_guard lock(central.mtx);
        central.bump = static_cast<char*>(raw);
        central.bump_end = central.bump + s;
        return;
      }
      const std::size_t pad = Upstream::alignment >= SizeClasses::kGranule ? 0 : SizeClasses::kGranule;
      const std::size_t bytes = nblocks * s + pad;
      void* raw = upstream_.allocate(bytes);  // may throw; no state modified yet
      if (raw == nullptr) throw std::bad_alloc{};
      {
        std::lock_guard lock(slabs_mtx_);
        try {
          slabs_.emplace_back(raw, bytes);
        } catch (...) {
          upstream_.deallocate(raw, bytes);
          throw;
        }
      }
      slab_bytes_.fetch_add(bytes, std::memory_order_relaxed);
      slab_count_.fetch_add(1, std::memory_order_relaxed);
      update_hwm();
      auto addr = reinterpret_cast<std::uintptr_t>(raw);
      addr = (addr + SizeClasses::kGranule - 1) & ~(std::uintptr_t{SizeClasses::kGranule} - 1);
      std::lock_guard lock(central.mtx);
      // only grow() sets bump, and we hold grow_mtx, so the bump region is still empty
      central.bump = reinterpret_cast<char*>(addr);
      central.bump_end = central.bump + nblocks * s;
    }

    void update_hwm() noexcept {
      const std::size_t r = reserved_bytes();
      std::size_t h = reserved_hwm_.load(std::memory_order_relaxed);
      while (r > h && !reserved_hwm_.compare_exchange_weak(h, r, std::memory_order_relaxed)) {
      }
    }

    void* allocate_large(std::size_t bytes) {
      void* p = upstream_.allocate(bytes);
      if (p == nullptr) throw std::bad_alloc{};
      large_bytes_.fetch_add(bytes, std::memory_order_relaxed);
      return p;
    }

    void deallocate_large(void* ptr, std::size_t bytes) noexcept {
      large_bytes_.fetch_sub(bytes, std::memory_order_relaxed);
      upstream_.deallocate(ptr, bytes);
    }

    Upstream upstream_;
    SizeClassPoolConfig cfg_;
    int num_classes_ = 0;
    std::size_t max_class_size_ = 0;
    std::unique_ptr<Central[]> centrals_;
    std::vector<std::size_t> class_size_;    ///< per-class block size
    std::vector<std::size_t> slab_nblocks_;  ///< per-class blocks per slab (>= 1); 1 means solo class
    std::vector<std::size_t> keep_;          ///< per-class max free blocks kept centrally
    std::vector<std::uint32_t> max_cached_;  ///< per-class thread cache capacity (may be 0)
    std::vector<std::uint32_t> batch_;       ///< per-class transfer batch size (>= 1)

    std::mutex slabs_mtx_;
    std::vector<std::pair<void*, std::size_t>> slabs_;
    std::atomic<std::size_t> slab_bytes_{0};
    std::atomic<std::size_t> slab_count_{0};
    std::atomic<std::size_t> solo_bytes_{0};
    std::atomic<std::size_t> solo_count_{0};
    std::atomic<std::size_t> reserved_hwm_{0};
    std::atomic<std::size_t> large_bytes_{0};
  };

  /**
   * Standard-conforming allocator over a singleton SizeClassPool.
   *
   * Like `umpire::default_init_allocator`, `construct(p)` default-initializes (so e.g.
   * `std::vector<double, A>::resize()` does not zero memory).
   *
   * @tparam PoolAccessor default-constructible callable returning a reference to the pool
   */
  template <typename T, typename PoolAccessor>
  class sizeclass_allocator {
    using pool_type = std::remove_reference_t<decltype(PoolAccessor{}())>;

   public:
    using value_type = T;
    using pointer = T*;
    using const_pointer = const T*;
    using reference = T&;              // still used by thrust
    using const_reference = const T&;  // ditto
    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;
    using is_always_equal = std::true_type;
    using propagate_on_container_move_assignment = std::true_type;

    template <typename U>
    struct rebind {
      using other = sizeclass_allocator<U, PoolAccessor>;
    };

    sizeclass_allocator() noexcept = default;
    template <typename U>
    sizeclass_allocator(const sizeclass_allocator<U, PoolAccessor>&) noexcept {}

    [[nodiscard]] T* allocate(std::size_t n) {
      static_assert(alignof(T) <= pool_type::alignment, "type is over-aligned for this pool");
      if (n > std::numeric_limits<std::size_t>::max() / sizeof(T)) throw std::bad_array_new_length{};
      return static_cast<T*>(PoolAccessor{}().allocate(n * sizeof(T)));
    }

    void deallocate(T* p, std::size_t n) noexcept { PoolAccessor{}().deallocate(p, n * sizeof(T)); }

    template <typename U>
    void construct(U* p) noexcept(std::is_nothrow_default_constructible_v<U>) {
      ::new (static_cast<void*>(p)) U;
    }
    template <typename U, typename... Args>
    void construct(U* p, Args&&... args) {
      ::new (static_cast<void*>(p)) U(std::forward<Args>(args)...);
    }

    template <typename U>
    friend bool operator==(const sizeclass_allocator&, const sizeclass_allocator<U, PoolAccessor>&) noexcept {
      return true;
    }
    template <typename U>
    friend bool operator!=(const sizeclass_allocator&, const sizeclass_allocator<U, PoolAccessor>&) noexcept {
      return false;
    }
  };

}  // namespace ttg::detail

#endif  // TTG_UTIL_SIZECLASS_POOL_H
