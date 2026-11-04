// SPDX-License-Identifier: BSD-3-Clause
//
// Pooled allocation of coroutine frames.
//
// Every TTG coroutine (device tasks, the coroutines that carry their sends,
// resumable host tasks) allocates its frame on the heap. These frames are short-lived,
// often larger than malloc's per-thread cache handles, and frequently freed by another
// thread than the one that allocated them (e.g., a send coroutine is created by the
// thread managing a device and destroyed by the worker completing the task), so plain
// operator new/delete puts malloc on the critical path of the device manager. The
// promise types inherit from pooled_coro_frame, which draws frames from a SizeClassPool
// with per-thread caches instead.

#ifndef TTG_UTIL_CORO_FRAME_POOL_H
#define TTG_UTIL_CORO_FRAME_POOL_H

#include <cstddef>
#include <cstdlib>
#include <new>

#include "ttg/util/sizeclass_pool.h"

namespace ttg::detail {

  /// upstream of the coroutine frame pool: cache-line aligned host memory from malloc
  struct coro_frame_upstream {
    static constexpr std::size_t alignment = 64;

    void* allocate(std::size_t bytes) {
      const std::size_t rounded = (bytes + alignment - 1) / alignment * alignment;
      void* ptr = std::aligned_alloc(alignment, rounded);
      if (ptr == nullptr) throw std::bad_alloc{};
      return ptr;
    }

    void deallocate(void* ptr, std::size_t) noexcept { std::free(ptr); }
  };

  using coro_frame_pool_t = SizeClassPool<coro_frame_upstream>;

  /// defaults of the coroutine frame pool: frames are a few hundred bytes to a few tens of KiB
  inline SizeClassPoolConfig coro_frame_pool_default_config() {
    SizeClassPoolConfig cfg;
    cfg.max_class_size = std::size_t{256} << 10;           // larger frames go to malloc directly
    cfg.slab_blocks = 64;
    cfg.min_slab_bytes = std::size_t{64} << 10;
    cfg.max_slab_bytes = std::size_t{1} << 20;
    cfg.central_keep_bytes = std::size_t{4} << 20;
    cfg.thread_cache_class_bytes = std::size_t{512} << 10;
    cfg.thread_cache_bytes = std::size_t{2} << 20;
    cfg.max_cached_blocks = 128;
    return cfg;
  }

  /// the process-wide coroutine frame pool; tunable via the environment variables
  /// TTG_CORO_POOL_{MAX_CLASS_BYTES, SLAB_BLOCKS, MIN_SLAB_BYTES, MAX_SLAB_BYTES, CENTRAL_KEEP_BYTES,
  /// THREAD_CACHE_CLASS_BYTES, THREAD_CACHE_BYTES, MAX_CACHED_BLOCKS} (see SizeClassPoolConfig)
  /// @note intentionally never destroyed, so that frames released and thread caches flushed at
  ///       thread exit (including after main() returns) always find a live pool
  inline coro_frame_pool_t& coro_frame_pool() {
    static coro_frame_pool_t* pool =
        new coro_frame_pool_t(SizeClassPoolConfig::from_env("TTG_CORO_POOL", coro_frame_pool_default_config()));
    return *pool;
  }

  /**
   * Base class of promise types whose coroutine frames are drawn from coro_frame_pool().
   * The compiler looks up operator new/delete in the scope of the promise type when
   * allocating a coroutine's frame, so inheriting from this class suffices.
   */
  struct pooled_coro_frame {
    static void* operator new(std::size_t size) { return coro_frame_pool().allocate(size); }

    static void operator delete(void* ptr, std::size_t size) noexcept { coro_frame_pool().deallocate(ptr, size); }

    /// frames with over-aligned locals: the pool hands out 64-byte aligned blocks
    static void* operator new(std::size_t size, std::align_val_t align) {
      if (static_cast<std::size_t>(align) <= coro_frame_pool_t::alignment) return coro_frame_pool().allocate(size);
      return ::operator new(size, align);
    }

    static void operator delete(void* ptr, std::size_t size, std::align_val_t align) noexcept {
      if (static_cast<std::size_t>(align) <= coro_frame_pool_t::alignment) {
        coro_frame_pool().deallocate(ptr, size);
      } else {
        ::operator delete(ptr, size, align);
      }
    }
  };

}  // namespace ttg::detail

#endif  // TTG_UTIL_CORO_FRAME_POOL_H
