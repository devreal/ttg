#ifndef TTG_PARSEC_MUTEXFLOW_H
#define TTG_PARSEC_MUTEXFLOW_H

#include <tuple>

#include "ttg/parsec/task.h"
#include "ttg/parsec/keyfns.h"

namespace ttg_parsec::detail {

    template<typename TT>
    struct mutexflow_elem {
      using key_type = ttg::meta::void_to_Void_t<typename TT::key_type>;

      using task_type = parsec_ttg_task_t<TT>;

    private:
      parsec_hash_table_item_t super;
      parsec_list_t m_tasks;                      // list of tasks waiting for a data
      parsec_ttg_task_base_t *m_active = nullptr; // currently active task
      ttg_data_copy_t *m_copy = nullptr;          // data waiting to be picked up by a task
      std::size_t m_count = 0;                    // number of tasks waiting for this data
      key_type m_key;                             // the key used to create the mutexflow
      std::unique_ptr<task_type> m_dummy_task;         // placeholder task to put into a copy

      task_type* make_dummy_task(TT* tt_ptr) {
        if constexpr (ttg::meta::is_void_v<key_type>) {
          return new task_type(nullptr, nullptr, nullptr, tt_ptr, 0); // dummy task
        } else {
          return new task_type(m_key, nullptr, nullptr, nullptr, tt_ptr, 0); // dummy task
        }
      }

    public:
      mutexflow_elem(const key_type& key, std::size_t count, ttg_data_copy_t* copy, TT* tt_ptr)
      : m_copy(copy)
      , m_key(key)
      , m_count(count)
      , m_dummy_task(make_dummy_task(tt_ptr)) // dummy task
      {
        super.key = reinterpret_cast<parsec_key_t>(&m_key);
        PARSEC_OBJ_CONSTRUCT(&m_tasks, parsec_list_t);
      }

      mutexflow_elem(task_type* task, std::size_t count)
      : m_key(task->key)
      , m_count(count)
      , m_dummy_task(make_dummy_task(task->tt)) // dummy task
      {
        super.key = reinterpret_cast<parsec_key_t>(&m_key);
        PARSEC_OBJ_CONSTRUCT(&m_tasks, parsec_list_t);
        parsec_list_nolock_push_back(&m_tasks, &task->parsec_task.super);
      }

      constexpr static ptrdiff_t offset_of_item() {
        return offsetof(mutexflow_elem<TT>, super);
      }

      void add_task(parsec_ttg_task_base_t* task) {
        assert(!parsec_list_nolock_contains(&m_tasks, &task->parsec_task.super));
        parsec_list_nolock_push_back(&m_tasks, &task->parsec_task.super);
      }

      void remove_task(parsec_ttg_task_base_t* task) {
        parsec_list_nolock_remove(&m_tasks, &task->parsec_task.super);
      }

      parsec_ttg_task_base_t* next_task() {
        parsec_ttg_task_base_t* task;
        task = (parsec_ttg_task_base_t*)parsec_list_nolock_pop_front(&m_tasks);
        return task;
      }

      bool has_task() const {
        return !parsec_list_nolock_is_empty(const_cast<parsec_list_t*>(&m_tasks));
      }

      bool has_copy() const {
        return nullptr != m_copy;
      }

      ttg_data_copy_t* get_copy() const {
        return m_copy;
      }

      void set_copy(ttg_data_copy_t* copy) {
        m_copy = copy;
      }

      void clear_copy() {
        m_copy = nullptr;
      }

      bool has_active_task() const {
        return m_active != nullptr;
      }

      parsec_ttg_task_base_t* get_active_task() const {
        return m_active;
      }

      void set_active_task(parsec_ttg_task_base_t* task) {
        m_active = task;
      }

      /**
       * Clear the currently active task and decrement the count of tasks in the bucket.
       * Returns the remaining count of tasks in the bucket.
       */
      std::size_t clear_active_task() {
        m_active = nullptr;
        return --m_count;
      }

      std::size_t get_count() const {
        return m_count;
      }

      parsec_hash_table_item_t& item() {
        return super;
      }

      task_type* dummy_task() {
        return m_dummy_task.get();
      }
    };

    template<typename Key, typename Value>
    struct finalizer_trait {
      using type = std::conditional_t<ttg::meta::is_void_v<Value>,
                                      std::function<void(const Key&)>,
                                      std::function<void(const Key&, const Value&)>>;
    };

    template<typename Value>
    struct finalizer_trait<ttg::Void, Value> {
      using type = std::conditional_t<ttg::meta::is_void_v<Value>,
                                      std::function<void()>,
                                      std::function<void(const Value&)>>;
    };
    template<typename Key, typename Value>
    using finalizer_type = finalizer_trait<Key, Value>::type;

    template<typename TT, std::size_t I>
    struct mutexflow {
    private:
      using keyT   = typename std::tuple_element_t<I, typename TT::input_terminals_type>::key_type;
      using valueT = typename std::tuple_element_t<I, typename TT::input_terminals_type>::value_type;

    public:
      using key_type    = ttg::meta::void_to_Void_t<keyT>;
      using value_type  = ttg::meta::void_to_Void_t<valueT>;
      using keymap_t    = ttg::meta::detail::keymap_t<key_type, key_type>;
      using countmap_t  = ttg::meta::detail::keymap_t<key_type, std::size_t>;
      using finalizer_t = finalizer_type<key_type, value_type>;

      using mutexflow_t  = mutexflow<TT, I>;
      using element_type = mutexflow_elem<TT>;


    private:

      struct parsec_hash_table_deleter {
        void operator()(parsec_hash_table_t *ht) const {
          if (ht) {
            parsec_hash_table_fini(ht);
            delete ht;
          }
        }
      };

      keymap_t m_map;
      countmap_t m_countmap;
      finalizer_t m_finalizer;
      std::unique_ptr<parsec_hash_table_t, parsec_hash_table_deleter> m_ht;

      static uint64_t key_hash(parsec_key_t k, void *user_data) {
        constexpr const bool keyT_is_Void = ttg::meta::is_void_v<keyT>;
        if constexpr (keyT_is_Void) {
          return 0;
        } else {
          mutexflow_t *flow = (mutexflow_t*)user_data;
          key_type &kk = *(reinterpret_cast<key_type *>(k));
          key_type bk = flow->map_key(kk); // map the key to a the bucket key
          parsec_key_t bk_ptr = reinterpret_cast<parsec_key_t>(&bk);
          return detail::key_hash<key_type>(bk_ptr, nullptr);
        }
      }

      static int key_equal(parsec_key_t a, parsec_key_t b, void *user_data) {
        constexpr const bool keyT_is_Void = ttg::meta::is_void_v<keyT>;
        if constexpr (keyT_is_Void) {
          return 1;
        } else {
          mutexflow_t *flow = (mutexflow_t*)user_data;
          key_type &ka = *(reinterpret_cast<key_type *>(a));
          key_type &kb = *(reinterpret_cast<key_type *>(b));
          key_type bka = flow->map_key(ka);
          key_type bkb = flow->map_key(kb);
          parsec_key_t bka_ptr = reinterpret_cast<parsec_key_t>(&bka);
          parsec_key_t bkb_ptr = reinterpret_cast<parsec_key_t>(&bkb);
          return detail::key_equal<key_type>(bka_ptr, bkb_ptr, nullptr);
        }
      }

      static const constexpr parsec_key_fn_t tasks_hash_fcts = {key_equal, detail::key_print<key_type>, key_hash};



    public:

      mutexflow() = default;

      template<typename MutexMap, typename CountMap, typename Finalizer>
      mutexflow(MutexMap&& map, CountMap&& countmap, Finalizer&& finalizer)
      : m_map(std::forward<MutexMap>(map))
      , m_countmap(std::forward<CountMap>(countmap))
      , m_finalizer(std::forward<Finalizer>(finalizer))
      , m_ht(new parsec_hash_table_t())
      {
        parsec_hash_table_init(m_ht.get(), element_type::offset_of_item(), 8,
                               tasks_hash_fcts, this);
      }

      /* mutexflow is move-only */
      mutexflow(const mutexflow& other) = delete;
      mutexflow(mutexflow&& other)
      : m_map(std::move(other.m_map))
      , m_countmap(std::move(other.m_countmap))
      , m_finalizer(std::move(other.m_finalizer))
      , m_ht(std::move(other.m_ht))
      {
        m_ht->hash_data = this; // set the user_data to this mutexflow instance
      }

      mutexflow& operator=(const mutexflow& other) = delete;
      mutexflow& operator=(mutexflow&& other) {
        if (this != &other) {
          m_map = std::move(other.m_map);
          m_countmap = std::move(other.m_countmap);
          m_finalizer = std::move(other.m_finalizer);
          m_ht = std::move(other.m_ht);
          m_ht->hash_data = this; // set the user_data to this mutexflow instance
        }
        return *this;
      }

      operator bool() const {
        return !!m_map;
      }

      key_type map_key(const key_type& key) const {
        constexpr const bool keyT_is_Void = ttg::meta::is_void_v<keyT>;
        if constexpr (keyT_is_Void) {
          return ttg::Void{};
        } else {
          return m_map(key);
        }
      }

      /**
       * Returns the number of elements in the bucket.
       */
      std::size_t get_count(const key_type& key) const {
        if constexpr (!ttg::meta::is_void_v<keyT>) {
          return m_countmap(key);
        } else {
          // if Value is not void, we assume the count is always 1
          return m_countmap();
        }
      }

      /**
       * Invokes the finalizer function.
       */
      template<typename ValueT>
      void finalize(const key_type& key, ValueT&& value) {
        auto invoke_finalizer = [&](auto... key) {
          if constexpr (ttg::meta::is_void_v<value_type>) {
            m_finalizer(key...);
          } else {
            m_finalizer(key..., value);
          }
        };
        if constexpr (ttg::meta::is_void_v<keyT>) {
          invoke_finalizer();
        } else {
          invoke_finalizer(key);
        }
      }

      /**
       * Returns a pointer to the stored hash table.
       * @note: This is a low-level function, use with care.
       */
      parsec_hash_table_t* get_hash_table() const {
        return m_ht.get();
      }
    };


    /**
     * A type trait to generate a tuple of mutexflow types based
     * on the provided key and input arguments.
     */
    template<typename TT, typename Indices>
    struct mutexflow_tuple
    { };

    template<typename TT, std::size_t... Is>
    struct mutexflow_tuple<TT, std::index_sequence<Is...>> {
      using type = std::tuple<detail::mutexflow<TT, Is>...>;
    };
    template<typename TT>
    using mutexflow_tuple_type = typename mutexflow_tuple<TT, std::make_index_sequence<TT::numinedges>>::type;

    template<typename Fn, typename... MutexFlows, std::size_t... Is>
    bool apply_mutexflow_with_index_helper(Fn&& fn, std::tuple<MutexFlows...>& mutexflows, std::index_sequence<Is...>) {
      // if any mutexflow handled the task, we can return early
      return (fn.template operator()<Is>(std::get<Is>(mutexflows)) || ...);
    }
    template<typename Fn, typename... MutexFlows>
    bool apply_mutexflow_with_index(Fn&& fn, std::tuple<MutexFlows...>& mutexflows) {
      if constexpr (sizeof...(MutexFlows) == 0) {
        return false; // no mutexflows to apply
      } else {
        return apply_mutexflow_with_index_helper(std::forward<Fn>(fn),
                                                 mutexflows,
                                                 std::index_sequence_for<MutexFlows...>{});
      }
    }

    template<typename Fn, typename... MutexFlows, std::size_t... Is>
    bool apply_mutexflow_helper(Fn&& fn, std::tuple<MutexFlows...>& mutexflows, std::index_sequence<Is...>) {
      // if any mutexflow handled the task, we can return early
      return (fn(std::get<Is>(mutexflows)) || ...);
    }
    template<typename Fn, typename... MutexFlows>
    bool apply_mutexflow(Fn&& fn, std::tuple<MutexFlows...>& mutexflows) {
      if constexpr (sizeof...(MutexFlows) == 0) {
        return false; // no mutexflows to apply
      } else {
        return apply_mutexflow_helper(std::forward<Fn>(fn),
                                      mutexflows,
                                      std::index_sequence_for<MutexFlows...>{});
      }
    }


} // namespace ttg_parsec::detail

#endif // TTG_PARSEC_MUTEXFLOW_H