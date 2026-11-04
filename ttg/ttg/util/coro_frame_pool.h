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

#include "ttg/util/pooled_object.h"

namespace ttg::detail {

  /// pool of coroutine frames: frames are a few hundred bytes to a few tens of KiB
  struct coro_frame_pool_tag {
    static constexpr const char* env_prefix = "TTG_CORO_POOL";

    static SizeClassPoolConfig default_config() {
      SizeClassPoolConfig cfg;
      cfg.max_class_size = std::size_t{256} << 10;  // larger frames go to malloc directly
      cfg.slab_blocks = 64;
      cfg.min_slab_bytes = std::size_t{64} << 10;
      cfg.max_slab_bytes = std::size_t{1} << 20;
      cfg.central_keep_bytes = std::size_t{4} << 20;
      cfg.thread_cache_class_bytes = std::size_t{512} << 10;
      cfg.thread_cache_bytes = std::size_t{2} << 20;
      cfg.max_cached_blocks = 128;
      return cfg;
    }
  };

  /// the process-wide coroutine frame pool, tunable via TTG_CORO_POOL_* environment variables
  inline object_pool_t<coro_frame_pool_tag>& coro_frame_pool() { return object_pool<coro_frame_pool_tag>(); }

  /**
   * Base class of promise types whose coroutine frames are drawn from coro_frame_pool().
   * The compiler looks up operator new/delete in the scope of the promise type when
   * allocating a coroutine's frame, so inheriting from this class suffices.
   */
  using pooled_coro_frame = pooled_object<coro_frame_pool_tag>;

}  // namespace ttg::detail

#endif  // TTG_UTIL_CORO_FRAME_POOL_H
