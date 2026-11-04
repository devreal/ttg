// SPDX-License-Identifier: BSD-3-Clause
#ifndef TTG_DATA_COPY_H
#define TTG_DATA_COPY_H

#include <utility>
#include <limits>
#include <vector>
#include <iterator>
#include <atomic>
#include <type_traits>

#if defined(PARSEC_HAVE_DEV_CUDA_SUPPORT)
#include <cuda_runtime.h>
#endif // PARSEC_HAVE_DEV_CUDA_SUPPORT

#include <parsec.h>

#include "ttg/parsec/thread_local.h"
#include "ttg/parsec/parsec-ext.h"
#include "ttg/util/span.h"
#include "ttg/util/pooled_object.h"


namespace ttg_parsec {

  namespace detail {

    // fwd-decl
    struct ttg_data_copy_t;

    /* templated to break cyclic dependency with ttg_data_copy_container */
    template<typename T = ttg_data_copy_t>
    struct ttg_data_copy_container_setter {
      ttg_data_copy_container_setter(T* ptr) {
        /* set the container ptr here, will be reset in the the ttg_data_value_copy_t ctor */
        ttg_data_copy_container() = ptr;
      }
    };

    /* special type: stores a pointer to the ttg_data_copy_t. This is necessary
     * because ttg_data_copy_t has virtual functions so we cannot cast from parsec_data_copy_t
     * to ttg_data_copy_t (offsetof is not supported for virtual classes).
     * The self pointer is a back-pointer to the ttg_data_copy_t. */
    struct ttg_data_copy_self_t {
      parsec_list_item_t super;
      ttg_data_copy_t *self;
      ttg_data_copy_self_t(ttg_data_copy_t* dc)
      : self(dc)
      {
        PARSEC_OBJ_CONSTRUCT(&super, parsec_list_item_t);
      }
    };

    /* Non-owning copy-tracking wrapper, accounting for N readers or 1 writer.
     * Also counts external references, which are not treated as
     * readers or writers but merely prevent the object from being
     * destroyed once no readers/writers exist.
     */
    /// pool of ttg_data_copy_t objects (TTValue-derived values and value wrappers)
    struct data_copy_pool_tag {
      static constexpr const char* env_prefix = "TTG_DATA_COPY_POOL";

      static ttg::detail::SizeClassPoolConfig default_config() {
        ttg::detail::SizeClassPoolConfig cfg;
        cfg.max_class_size = std::size_t{64} << 10;  // larger values go to malloc directly
        cfg.slab_blocks = 64;
        cfg.min_slab_bytes = std::size_t{64} << 10;
        cfg.max_slab_bytes = std::size_t{1} << 20;
        cfg.central_keep_bytes = std::size_t{4} << 20;
        cfg.thread_cache_class_bytes = std::size_t{256} << 10;
        cfg.thread_cache_bytes = std::size_t{2} << 20;
        cfg.max_cached_blocks = 128;
        return cfg;
      }
    };

    /* Copies are created by whichever thread sends a value that is not tracked yet (for
     * device tasks: the thread managing the device) and released by the thread releasing
     * the last reference, so they are allocated from a pool with per-thread caches, which
     * TTValue-derived types inherit as well (see ttg::detail::pooled_object). Copies are
     * always deleted through ttg_data_copy_t*, whose destructor is virtual, so the pool
     * receives the size of the dynamic type. Placement new is supported for values
     * constructed in place, e.g. when deserializing. */
    struct ttg_data_copy_t : public ttg_data_copy_self_t
                           , public ttg::detail::pooled_object_with_placement<data_copy_pool_tag> {

      /* special value assigned to parsec_data_copy_t::readers to mark the copy as
      * mutable, i.e., a task will modify it */
      static constexpr int mutable_tag = std::numeric_limits<int>::min();

      ttg_data_copy_t()
      : ttg_data_copy_self_t(this)
      { }

      ttg_data_copy_t(const ttg_data_copy_t& c)
      : ttg_data_copy_self_t(this)
      {
        /* we allow copying but do not copy any data over from the original
         * device copies will have to be allocated again
         * and it's a new object to reference */
      }

      ttg_data_copy_t(ttg_data_copy_t&& c)
      : ttg_data_copy_self_t(this)
      , m_next_task(c.m_next_task)
      , m_readers(c.m_readers)
      , m_refs(c.m_refs.load(std::memory_order_relaxed))
      {
        c.m_readers = 0;
      }

      ttg_data_copy_t& operator=(ttg_data_copy_t&& c)
      {
        m_next_task = c.m_next_task;
        c.m_next_task = nullptr;
        m_readers = c.m_readers;
        c.m_readers = 0;
        m_refs.store(c.m_refs.load(std::memory_order_relaxed), std::memory_order_relaxed);
        c.m_refs.store(0, std::memory_order_relaxed);
        return *this;
      }

      ttg_data_copy_t& operator=(const ttg_data_copy_t& c) {
        /* we allow copying but do not copy any data over from the original
         * device copies will have to be allocated again
         * and it's a new object to reference */

        return *this;
      }

      /* mark destructor as virtual */
      virtual ~ttg_data_copy_t() = default;

      /* Returns true if the copy is mutable */
      bool is_mutable() const {
        return m_readers == mutable_tag;
      }

      /* Mark the copy as mutable */
      void mark_mutable() {
        m_readers = mutable_tag;
      }

      /* Increment the reader counter and return previous value
      * \tparam Atomic Whether to decrement atomically. Default: true
      */
      template<bool Atomic = true>
      int increment_readers() {
        if constexpr(Atomic) {
          return parsec_atomic_fetch_inc_int32(&m_readers);
//          std::atomic_ref<int32_t> a{m_readers};
//          return a.fetch_add(1, std::memory_order_relaxed);
        } else {
          return m_readers++;
        }
      }

      /**
      * Reset the number of readers to read-only with a single reader.
      */
      void reset_readers() {
        if (mutable_tag == m_readers) {
          m_readers = 1;
        }
      }

      /* Decrement the reader counter and return previous value.
      * \tparam Atomic Whether to decrement atomically. Default: true
      */
      template<bool Atomic = true>
      int decrement_readers() {
        if constexpr(Atomic) {
          return parsec_atomic_fetch_dec_int32(&m_readers);
//          std::atomic_ref<int32_t> a{m_readers};
//          return a.fetch_sub(1, std::memory_order_relaxed);
        } else {
          return m_readers--;
        }
      }

      /* Returns the number of readers if the copy is immutable, or \c mutable_tag
      * if the copy is mutable */
      int num_readers() const {
        return m_readers;
      }

      /* Returns the pointer to the user data wrapped by the the copy object */
      virtual void* get_ptr() = 0;

      parsec_task_t* get_next_task() const {
        return m_next_task;
      }

      void set_next_task(parsec_task_t* task) {
        m_next_task = task;
      }

      int32_t add_ref() {
        return m_refs.fetch_add(1, std::memory_order_relaxed);
      }

      int32_t drop_ref() {
        return m_refs.fetch_sub(1, std::memory_order_relaxed);
      }

      bool has_ref() {
        return (m_refs.load(std::memory_order_relaxed) != 0);
      }

      int32_t num_ref() const {
        return m_refs.load(std::memory_order_relaxed);
      }

#if defined(PARSEC_PROF_TRACE) && defined(PARSEC_TTG_PROFILE_BACKEND)
      int64_t size;
      int64_t uid;
#endif
    protected:
      parsec_task_t *m_next_task = nullptr;
      int32_t        m_readers  = 1;
      std::atomic<int32_t>  m_refs = 1;                     //< number of entities referencing this copy (TTGs, external)
    };


    /**
    * Extension of ttg_data_copy_t holding the actual value.
    * The virtual destructor will take care of destructing the value if
    * the destructor of ttg_data_copy_t base class is called.
    */
    template<typename ValueT>
    struct ttg_data_value_copy_t final : private ttg_data_copy_container_setter<ttg_data_copy_t>
                                       , public ttg_data_copy_t {
      using value_type = ValueT;
      value_type m_value;

      template<typename T>
      requires(std::constructible_from<ValueT, T>)
      ttg_data_value_copy_t(T&& value)
      : ttg_data_copy_container_setter(this)
      , ttg_data_copy_t()
      , m_value(std::forward<T>(value))
      {
        /* reset the container tracker */
        ttg_data_copy_container() = nullptr;
      }

      ttg_data_value_copy_t(ttg_data_value_copy_t&& c)
        noexcept(std::is_nothrow_move_constructible_v<value_type>)
      : ttg_data_copy_container_setter(this)
      , ttg_data_copy_t(std::move(c))
      , m_value(std::move(c.m_value))
      {
        /* reset the container tracker */
        ttg_data_copy_container() = nullptr;
      }

      ttg_data_value_copy_t(const ttg_data_value_copy_t& c)
        noexcept(std::is_nothrow_copy_constructible_v<value_type>)
      : ttg_data_copy_container_setter(this)
      , ttg_data_copy_t(c)
      , m_value(c.m_value)
      {
        /* reset the container tracker */
        ttg_data_copy_container() = nullptr;
      }

      ttg_data_value_copy_t& operator=(ttg_data_value_copy_t&& c)
        noexcept(std::is_nothrow_move_assignable_v<value_type>)
      {
        /* set the container ptr here, will be reset in the the ttg_data_value_copy_t ctor */
        ttg_data_copy_container() = this;
        ttg_data_copy_t::operator=(std::move(c));
        m_value = std::move(c.m_value);
        /* reset the container tracker */
        ttg_data_copy_container() = nullptr;
      }

      ttg_data_value_copy_t& operator=(const ttg_data_value_copy_t& c)
        noexcept(std::is_nothrow_copy_assignable_v<value_type>)
      {
        /* set the container ptr here, will be reset in the the ttg_data_value_copy_t ctor */
        ttg_data_copy_container() = this;
        ttg_data_copy_t::operator=(c);
        m_value = c.m_value;
        /* reset the container tracker */
        ttg_data_copy_container() = nullptr;
      }

      value_type& operator*() {
        return m_value;
      }

      /* will destruct the value */
      virtual ~ttg_data_value_copy_t() = default;

      virtual void* get_ptr() override final {
        return &m_value;
      }
    };
  } // namespace detail

} // namespace ttg_parsec

#endif // TTG_DATA_COPY_H
