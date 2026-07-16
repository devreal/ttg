// SPDX-License-Identifier: BSD-3-Clause
#ifndef TTG_DEVICE_BATCH_H
#define TTG_DEVICE_BATCH_H

#include <cstddef>
#include <tuple>
#include <type_traits>
#include <vector>

#ifdef TTG_HAVE_COROUTINE

namespace ttg::device {

  namespace detail {
    /* tag type produced by ttg::device::coop(); consumed by
     * device_task_promise_type::await_transform (see ttg/device/task.h) */
    template <typename Key, typename... Args>
    struct coop_t {
      std::tuple<Args&...> args;
    };
  }  // namespace detail

  /**
   * One member of a batch collected via \sa ttg::device::coop.
   * Exposes the arguments that member passed to its own \c coop() call.
   */
  template <typename Key, typename... Args>
  class batch_member {
   public:
    batch_member(std::tuple<Args&...>& args) : m_args(&args) {}

    /// @return the tuple of arguments this member passed to \c coop()
    std::tuple<Args&...>& args() const { return *m_args; }

    /// @return the @c I-th argument this member passed to \c coop()
    template <std::size_t I>
    decltype(auto) get() const {
      return std::get<I>(*m_args);
    }

   private:
    std::tuple<Args&...>* m_args;
  };

  /**
   * A batch of sibling tasks collected together by the runtime via
   * \sa ttg::device::coop, based on the compatibility predicate provided to
   * \c TT::set_batch_matcher. Exactly one member (\sa is_leader) is
   * responsible for issuing the (possibly batched) kernel launch using the
   * arguments of every member (\sa begin / end / operator[]); all members,
   * leader and followers alike, proceed identically afterwards (typically
   * `co_await ttg::device::wait(...)`).
   *
   * If no siblings were collected (batching disabled or unsupported by the
   * runtime, or no compatible candidates were ready), the batch has
   * `size() == 1` and `is_leader() == true`, so the same application code
   * works whether or not batching actually occurred.
   */
  template <typename Key, typename... Args>
  class batch_view {
   public:
    using member_type = batch_member<Key, Args...>;

    batch_view() = default;
    batch_view(std::vector<member_type> members, bool is_leader)
        : m_members(std::move(members)), m_is_leader(is_leader) {}

    /// @return true if this task is responsible for submitting the (possibly
    ///         batched) kernel on behalf of the whole batch
    bool is_leader() const { return m_is_leader; }

    /// @return the number of tasks collected into this batch (always >= 1)
    std::size_t size() const { return m_members.size(); }

    auto begin() const { return m_members.begin(); }
    auto end() const { return m_members.end(); }

    const member_type& operator[](std::size_t i) const { return m_members[i]; }

   private:
    std::vector<member_type> m_members;
    bool m_is_leader = false;
  };

  /**
   * Collect the batch of sibling tasks the runtime formed together with this
   * task (\sa TT::set_batch_matcher). Must be awaited immediately after
   * \sa ttg::device::select has resumed. The provided \c args are exposed to
   * every member of the returned batch (including this task itself) via
   * \sa batch_member::args / batch_member::get -- this is how data that is
   * not exposed through \c select (e.g., per-task scalars) can be shared
   * with whichever member ends up submitting the batched kernel.
   *
   * \tparam Key the task's key type (must be specified explicitly; used only
   *         to tag the returned \c batch_view type)
   */
  template <typename Key, typename... Args>
  [[nodiscard]] inline auto coop(Args&&... args) {
    return detail::coop_t<Key, std::remove_reference_t<Args>...>{std::tie(std::forward<Args>(args)...)};
  }

}  // namespace ttg::device

#endif  // TTG_HAVE_COROUTINE

#endif  // TTG_DEVICE_BATCH_H
