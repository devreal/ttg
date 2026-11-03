// SPDX-License-Identifier: BSD-3-Clause
//
// Created by Eduard Valeyev on 5/20/25.
//

#ifndef TTG_ENV_H
#define TTG_ENV_H

#include "ttg/config.h"
#include "ttg/util/sizeclass_pool.h"

#include <umpire_cxx_allocator.hpp>

#include <umpire/strategy/QuickPool.hpp>
#include <umpire/strategy/SizeLimiter.hpp>

#include <cassert>
#include <memory>
#include <mutex>

namespace ttg {

  /**
   * Env maintains the runtime environment of TTG,
   * such as device artifacts, memory allocators, etc.
   *
   * \note this is a Singleton
   */
  class Env {
   public:
    ~Env() {
    }

    Env(const Env&) = delete;
    Env(Env&&) = delete;
    Env& operator=(const Env&) = delete;
    Env& operator=(Env&&) = delete;

    /// access the singleton instance; if not initialized will be
    /// initialized via Env::initialize() with the default params
    static std::unique_ptr<Env>& instance() {
      if (!instance_accessor()) {
        initialize();
      }
      return instance_accessor();
    }

    // clang-format off
    /// initialize the instance using explicit params
    /// \param world the world to use for initialization
    /// \param page_size memory added to the pools supporting `this->um_allocator()`, `this->device_allocator()`, and `this->pinned_allocator()` in chunks of at least
    ///                  this size (bytes) [default=2^25]
    /// \param pinned_alloc_limit the maximum total amount of memory (in bytes) that
    ///        allocator returned by `this->pinned_allocator()` can allocate [default=2^40]
    // clang-format on
    static void initialize(const std::uint64_t page_size = (1ul << 25),
                           const std::uint64_t pinned_alloc_limit = (1ul << 40)) {
      static std::mutex mtx;  // to make initialize() reentrant
      std::scoped_lock lock{mtx};
      // only the winner of the lock race gets to initialize
      if (instance_accessor() == nullptr) {

        // uncomment to debug umpire ops
        //
        //      umpire::util::Logger::getActiveLogger()->setLoggingMsgLevel(
        //          umpire::util::message::Debug);

        auto& rm = umpire::ResourceManager::getInstance();

        // turn off Umpire introspection for non-Debug builds
#ifndef NDEBUG
        constexpr auto introspect = true;
#else
        constexpr auto introspect = false;
#endif

        // allocate pinned_alloc_limit in pinned memory
        auto pinned_size_limited_alloc =
            rm.makeAllocator<umpire::strategy::SizeLimiter, introspect>(
                "SizeLimited_PINNED", rm.getAllocator("PINNED"),
                pinned_alloc_limit);
        auto pinned_dynamic_pool =
            rm.makeAllocator<umpire::strategy::QuickPool, introspect>(
                "QuickPool_SizeLimited_PINNED", pinned_size_limited_alloc,
                /* first_minimum_pool_allocation_size = */ 0,
                /* next_minimum_pool_allocation_size = */ page_size,
                /* alignment */ 16);

        auto env = std::unique_ptr<Env>(new Env(pinned_dynamic_pool));
        instance_accessor() = std::move(env);
      }
    }

    /// @return an Umpire allocator that allocates from a
    ///         pinned memory pool
    /// @warning this is not a thread-safe allocator, should be only used when
    ///          wrapped into umpire_based_allocator_impl
    auto& pinned_allocator() { return pinned_allocator_; }

    // clang-format off
    /// @return the max actual amount of memory held by pinned_allocator()
    /// @details returns the value provided by `umpire::strategy::QuickPool::getHighWatermark()`
    /// @note if there is only 1 Umpire allocator using PINNED memory this should be identical to the value returned by `umpire::ResourceManager::getInstance().getAllocator("PINNED").getHighWatermark()`
    // clang-format on
    std::size_t pinned_allocator_getActualHighWatermark() {
      assert(dynamic_cast<::umpire::strategy::QuickPool*>(
                    pinned_allocator_.getAllocationStrategy()) != nullptr);
      return dynamic_cast<::umpire::strategy::QuickPool*>(
                 pinned_allocator_.getAllocationStrategy())
          ->getActualHighwaterMark();
    }

    std::size_t pinned_allocator_getCurrentSize() {
      assert(dynamic_cast<::umpire::strategy::QuickPool*>(
                    pinned_allocator_.getAllocationStrategy()) != nullptr);
      return dynamic_cast<::umpire::strategy::QuickPool*>(
                 pinned_allocator_.getAllocationStrategy())
          ->getCurrentSize();
    }


   protected:
    Env(::umpire::Allocator pinned_alloc)
        : pinned_allocator_(pinned_alloc) {}

   private:
    // allocates from a dynamic, size-limited pinned memory pool
    // N.B. not thread safe, so must be wrapped into umpire_based_allocator_impl
    ::umpire::Allocator pinned_allocator_;

    inline static std::unique_ptr<Env>& instance_accessor() {
      static std::unique_ptr<Env> instance_{nullptr};
      return instance_;
    }
  };  // class Env

  namespace umpire {

    struct pinned_allocator_getter {
      inline ::umpire::Allocator& operator()() { return Env::instance()->pinned_allocator(); }
    };

  }

  namespace detail {

    /// Upstream of the size-class pinned pool: obtains slabs (and oversized blocks) from
    /// Env::pinned_allocator(). Uses the same lock as umpire_pinned_allocator_t, so both
    /// allocators can be used side by side.
    struct pinned_pool_upstream {
      /// QuickPool in Env::initialize() is created with 16-byte alignment
      static constexpr std::size_t alignment = 16;

      void* allocate(std::size_t bytes) {
        auto* strategy = Env::instance()->pinned_allocator().getAllocationStrategy();
        env_lock_guard lock;
        return strategy->allocate_internal(bytes);
      }

      void deallocate(void* ptr, std::size_t bytes) {
        auto* strategy = Env::instance()->pinned_allocator().getAllocationStrategy();
        env_lock_guard lock;
        strategy->deallocate_internal(ptr, bytes);
      }

     private:
      /// holds the lock that also serializes umpire_pinned_allocator_t
      struct env_lock_guard {
        env_lock_guard() { ::umpire::detail::MutexLock<::ttg::Env>::lock(); }
        ~env_lock_guard() { ::umpire::detail::MutexLock<::ttg::Env>::unlock(); }
        env_lock_guard(const env_lock_guard&) = delete;
        env_lock_guard& operator=(const env_lock_guard&) = delete;
      };
    };

    using pinned_pool_t = SizeClassPool<pinned_pool_upstream>;

    /// the process-wide size-class pinned pool; tunable via the environment variables
    /// TTG_PINNED_POOL_{MAX_CLASS_BYTES, SLAB_BLOCKS, MIN_SLAB_BYTES, MAX_SLAB_BYTES, CENTRAL_KEEP_BYTES,
    /// THREAD_CACHE_CLASS_BYTES, THREAD_CACHE_BYTES, MAX_CACHED_BLOCKS} (see SizeClassPoolConfig)
    /// @note intentionally never destroyed, so that thread caches flushed at thread exit
    ///       (including after main() returns) always find a live pool
    inline pinned_pool_t& pinned_pool() {
      static pinned_pool_t* pool = new pinned_pool_t(SizeClassPoolConfig::from_env("TTG_PINNED_POOL"));
      return *pool;
    }

    struct pinned_pool_getter {
      pinned_pool_t& operator()() const { return pinned_pool(); }
    };

  }  // namespace detail

  /// pooled thread-safe pinned host memory allocator, serialized by a single global mutex
  template <typename T>
  using umpire_pinned_allocator_t = ::umpire::default_init_allocator<
      T, ::umpire::allocator<T, ::umpire::detail::MutexLock<::ttg::Env>,
                             umpire::pinned_allocator_getter>>;

  /// pooled thread-scalable pinned host memory allocator (size classes + per-thread caches);
  /// draws slabs from the same pinned QuickPool as umpire_pinned_allocator_t
  /// @note memory, once obtained from the QuickPool, is retained by the pool for the lifetime of the
  ///       process; Env::pinned_allocator_getCurrentSize() reports it as in use
  template <typename T>
  using sizeclass_pinned_allocator_t = detail::sizeclass_allocator<T, detail::pinned_pool_getter>;

  /// pooled thread-safe pinned host memory allocator for device computing
#if defined(TTG_PINNED_ALLOCATOR_SIZECLASS)
  template <typename T>
  using pinned_allocator_t = sizeclass_pinned_allocator_t<T>;
#else
  template <typename T>
  using pinned_allocator_t = umpire_pinned_allocator_t<T>;
#endif

}  // namespace ttg

#if defined(__has_include)
#if __has_include(<madness/world/archive.h>)
#include <madness/world/archive.h>
namespace madness::archive {
  // sizeclass_allocator is stateless: nothing to (de)serialize
  template <class Archive, typename T, typename PoolAccessor>
  struct ArchiveLoadImpl<Archive, ttg::detail::sizeclass_allocator<T, PoolAccessor>> {
    static inline void load(const Archive&, ttg::detail::sizeclass_allocator<T, PoolAccessor>&) {}
  };
  template <class Archive, typename T, typename PoolAccessor>
  struct ArchiveStoreImpl<Archive, ttg::detail::sizeclass_allocator<T, PoolAccessor>> {
    static inline void store(const Archive&, const ttg::detail::sizeclass_allocator<T, PoolAccessor>&) {}
  };
}  // namespace madness::archive
#endif
#endif

#endif  // TTG_ENV_H
