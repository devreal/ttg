// SPDX-License-Identifier: BSD-3-Clause
//
// Class-scope allocation of objects from a SizeClassPool.
//
// A class inheriting from pooled_object<Tag> (directly or through a base class) is
// allocated by `new T(...)` from a process-wide SizeClassPool, one per Tag, and returned
// to it by `delete`. The pool has per-thread caches, so objects allocated by one thread
// and freed by another (e.g., data created by the thread managing a device and released
// by a worker) do not contend on malloc's arenas. Objects of such classes that are not
// created by a new-expression (automatic variables, members, container elements) are
// not affected.
//
// Each Tag provides:
//   static constexpr const char* env_prefix;       // prefix of the tuning environment variables
//   static SizeClassPoolConfig default_config();   // defaults for the pool's configuration

#ifndef TTG_UTIL_POOLED_OBJECT_H
#define TTG_UTIL_POOLED_OBJECT_H

#include <cstddef>
#include <cstdlib>
#include <new>

#include "ttg/util/sizeclass_pool.h"

namespace ttg::detail {

  /// upstream of the object pools: cache-line aligned host memory from malloc
  struct pooled_object_upstream {
    static constexpr std::size_t alignment = 64;

    void* allocate(std::size_t bytes) {
      const std::size_t rounded = (bytes + alignment - 1) / alignment * alignment;
      void* ptr = std::aligned_alloc(alignment, rounded);
      if (ptr == nullptr) throw std::bad_alloc{};
      return ptr;
    }

    void deallocate(void* ptr, std::size_t) noexcept { std::free(ptr); }
  };

  template <typename Tag>
  using object_pool_t = SizeClassPool<pooled_object_upstream, Tag>;

  /// the process-wide pool of objects of @p Tag; tunable via the environment variables
  /// <Tag::env_prefix>_{MAX_CLASS_BYTES, SLAB_BLOCKS, MIN_SLAB_BYTES, MAX_SLAB_BYTES, CENTRAL_KEEP_BYTES,
  /// THREAD_CACHE_CLASS_BYTES, THREAD_CACHE_BYTES, MAX_CACHED_BLOCKS} (see SizeClassPoolConfig)
  /// @note intentionally never destroyed, so that objects released and thread caches flushed at
  ///       thread exit (including after main() returns) always find a live pool
  template <typename Tag>
  inline object_pool_t<Tag>& object_pool() {
    static object_pool_t<Tag>* pool =
        new object_pool_t<Tag>(SizeClassPoolConfig::from_env(Tag::env_prefix, Tag::default_config()));
    return *pool;
  }

  /**
   * Base class of classes whose objects are allocated from object_pool<Tag>(). The
   * compiler looks up operator new/delete in the scope of the class of a new-expression
   * (and of a coroutine's promise type) first, so inheriting from this class suffices.
   *
   * @note Sized deallocation requires that objects are deleted through a pointer to their
   *       dynamic type, or through a base class with a virtual destructor.
   * @note Class-scope operator new hides the global ones, so placement new (`new (ptr) T`)
   *       and `new (std::nothrow) T` do not compile for such classes. Use
   *       pooled_object_with_placement for classes that are constructed with placement new.
   *       Promise types must not use it: coroutine frames are allocated by
   *       operator new(size, <coroutine arguments>...) if it is viable, which the placement
   *       form would be for a coroutine taking a single pointer.
   */
  template <typename Tag>
  struct pooled_object {
    static void* operator new(std::size_t size) { return object_pool<Tag>().allocate(size); }

    static void operator delete(void* ptr, std::size_t size) noexcept { object_pool<Tag>().deallocate(ptr, size); }

    /// over-aligned types: the pool hands out 64-byte aligned blocks
    static void* operator new(std::size_t size, std::align_val_t align) {
      if (static_cast<std::size_t>(align) <= object_pool_t<Tag>::alignment) return object_pool<Tag>().allocate(size);
      return ::operator new(size, align);
    }

    static void operator delete(void* ptr, std::size_t size, std::align_val_t align) noexcept {
      if (static_cast<std::size_t>(align) <= object_pool_t<Tag>::alignment) {
        object_pool<Tag>().deallocate(ptr, size);
      } else {
        ::operator delete(ptr, size, align);
      }
    }
  };

  /**
   * pooled_object that also supports placement new (`new (ptr) T`), for classes that are
   * constructed in storage provided by the caller, e.g. when deserializing.
   * @note `new (std::nothrow) T` is still not supported.
   */
  template <typename Tag>
  struct pooled_object_with_placement : pooled_object<Tag> {
    using pooled_object<Tag>::operator new;
    using pooled_object<Tag>::operator delete;

    /// placement new: storage provided by the caller, nothing to release
    static void* operator new(std::size_t, void* ptr) noexcept { return ptr; }
    static void operator delete(void*, void*) noexcept {}
  };

}  // namespace ttg::detail

#endif  // TTG_UTIL_POOLED_OBJECT_H
