#ifndef TTG_MADNESS_FWD_H
#define TTG_MADNESS_FWD_H

#include "ttg/fwd.h"

#include <future>

namespace ttg_madness {

  template <typename keyT, typename output_terminalsT, typename derivedT, typename... input_valueTs>
  class Op;

  class WorldImpl;

  template <typename... RestOfArgs>
  inline void ttg_initialize(int argc, char **argv, RestOfArgs &&...);

  inline void ttg_finalize();

  inline void ttg_abort();

  inline ttg::World ttg_default_execution_context();

  inline void ttg_execute(ttg::World world);

  inline void ttg_fence(ttg::World world);

  template <typename T>
  inline void ttg_register_ptr(ttg::World world, const std::shared_ptr<T> &ptr);

  inline void ttg_register_status(ttg::World world, const std::shared_ptr<std::promise<void>> &status_ptr);

  inline ttg::Edge<> &ttg_ctl_edge(ttg::World world);

  template <typename T>
  inline void ttg_sum(ttg::World world, T &value);

  template <typename T>
  inline void ttg_broadcast(ttg::World world, T &data, int source_rank);

}  // namespace ttg_madness

#endif  // TTG_MADNESS_FWD_H
