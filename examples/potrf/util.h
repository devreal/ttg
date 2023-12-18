#pragma once

#include <cmath>

#if defined(TTG_HAVE_CUDART)
#define ES ttg::ExecutionSpace::CUDA
#define TASKRET -> ttg::device_task
#include <cusolverDn.h>
#elif defined(TTG_HAVE_HIP)
#define ES ttg::ExecutionSpace::HIP
#define TASKRET -> ttg::device_task
#include <hipsolver/hipsolver.h>
#include <hipblas/hipblas.h>
#else
#define ES ttg::ExecutionSpace::Host
#define TASKRET -> void
#endif

inline auto check_norm(double expected, double actual) {
  if (std::abs(expected - actual) <= std::max(std::abs(expected), std::abs(actual))*1E-12) {
    return true;
  }
  return false;
}

#ifdef TTG_HAVE_KOKKOS
void kokkos_init(int& argc, char* argv[]);
void kokkos_finalize();
#else  // TTG_HAVE_KOKKOS
void kokkos_init(int& argc, char* argv[]) { }
void kokkos_finalize() { }
#endif // TTG_HAVE_KOKKOS