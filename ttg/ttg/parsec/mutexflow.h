#ifndef TTG_PARSEC_MUTEXFLOW_H
#define TTG_PARSEC_MUTEXFLOW_H

#include "ttg/parsec/task.h"

namespace ttg_parsec::detail {

    template<typename Key>
    struct mutexflow_elem {
    private:
      parsec_hash_table_item_t tt_ht_item;
      parsec_list_t m_tasks;              // list of tasks waiting for a data
      parsec_ttg_task_base_t *m_active = nullptr; // currently active task
      ttg_data_copy_t *m_copy = nullptr;  // data waiting to be picked up by a task
      Key m_key;
      std::size_t m_count;

    public:
      mutexflow_elem(const Key& key, ttg_data_copy_t* copy, std::size_t count)
      : m_copy(copy)
      , m_key(key)
      , m_count(count)
      { }

      mutexflow_elem(const Key& key, parsec_task_t* task, std::size_t count)
      : m_key(key)
      , m_count(count)
      {
        PARSEC_OBJ_CONSTRUCT(&m_tasks, parsec_list_t);
        parsec_list_nolock_push_back(&m_tasks, task);
      }

      void add_task(parsec_ttg_task_base_t* task) {
        parsec_list_nolock_push_back(&m_tasks, &task->parsec_task.super);
      }

      void remove_task(parsec_ttg_task_base_t* task) {
        parsec_list_nolock_remove(&m_tasks, task);
      }

      parsec_ttg_task_base_t* next_task() {
        parsec_ttg_task_base_t* task;
        task = (parsec_ttg_task_base_t*)parsec_list_nolock_pop_front(&m_tasks);
        return task;
      }

      bool has_tasks() const {
        return !parsec_list_nolock_is_empty(&m_tasks);
      }

      bool has_copy() const {
        return nullptr != m_copy;
      }

      ttg_data_copy_t* get_copy() const {
        return m_copy;
      }

      parsec_ttg_task_base_t* get_active_task() const {
        return m_active;
      }

      void set_active_task(parsec_ttg_task_base_t* task) {
        m_active = task;
      }
    };

    template<typename keyT>
    struct mutexflow {
      ttg::meta::detail::keymap_t<keyT, uint64_t> m_map;
      ttg::meta::detail::keymap_t<uint64_t, uint64_t> m_count;
      parsec_hash_table_t m_ht;

      using mutexflow_t  = mutexflow<keyT>;
      using mutexflow_elem_t = mutexflow_elem<keyT>;

    private:

      static uint64_t key_hash(parsec_key_t k, void *user_data) {
        constexpr const bool keyT_is_Void = ttg::meta::is_void_v<keyT>;
        if constexpr (keyT_is_Void || std::is_same_v<keyT, void>) {
          return 0;
        } else {
          mutexflow_t *flow = (mutexflow_t*)user_data;
          keyT &kk = *(reinterpret_cast<keyT *>(k));
          uint64_t hv = flow->m_map(kk);
          return hv;
        }
      }

      static int key_equal(parsec_key_t a, parsec_key_t b, void *user_data) {
        if constexpr (std::is_same_v<keyT, void>) {
          return 1;
        } else {
          mutexflow_t *flow = (mutexflow_t*)user_data;
          keyT &ka = *(reinterpret_cast<keyT *>(a));
          keyT &kb = *(reinterpret_cast<keyT *>(b));
          return flow->m_map(ka) == flow->m_map(kb);
        }
      }

      static const constexpr parsec_key_fn_t tasks_hash_fcts = {key_equal, detail::key_print<keyT>, key_hash};

    public:

      mutexflow() = default;
      template<typename MutexMap, typename CountMap>
      mutexflow(MutexMap&& map, CountMap&& count)
      : m_map(std::forward<MutexMap>(map))
      , m_count(std::forward<CountMap>(count))
      {
        parsec_hash_table_init(&m_ht, offsetof(mutexflow_elem<keyT>, tt_ht_item), 8,
                               tasks_hash_fcts, this);
      }

      ~mutexflow() {
        parsec_hash_table_fini(&m_ht);
      }

      operator bool() const {
        return !!m_map;
      }
    };
}

#endif // TTG_PARSEC_MUTEXFLOW_H