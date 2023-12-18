
#include <Kokkos_Core.hpp>

//#include "ttg/device/device.h"

#include <iostream>


// fwd-decl
namespace ttg::device {
cudaStream_t current_stream();
} // namespace ttg::device

//#define KOKKOS_PLGSY_HOST

#ifdef KOKKOS_PLGSY_HOST
#define KOKKOS_SPACE Kokkos::HostSpace
#define KOKKOS_POLICY Kokkos::Serial
#define KOKKOS_POLICY_INSTANCE Kokkos::Serial()
#else
#define KOKKOS_SPACE Kokkos::Cuda::memory_space
#define KOKKOS_POLICY Kokkos::Cuda
#define KOKKOS_POLICY_INSTANCE Kokkos::Cuda(ttg::device::current_stream())
#endif

#include "random.h"

namespace detail {

  template<typename T>
  struct num_elems : std::integral_constant<int, 1>
  { };

  template<typename T>
  struct num_elems<std::complex<T>> : std::integral_constant<int, 2>
  { };

  template<typename T>
  static constexpr int num_elems_v = num_elems<T>::value;

  template<typename T>
  struct is_complex : std::integral_constant<bool, false>
  { };

  template<typename T>
  struct is_complex<std::complex<T>> : std::integral_constant<bool, true>
  { };

  template<typename T>
  static constexpr bool is_complex_v = is_complex<T>::value;
} // namespace detail

template <typename T>
void CORE_plgsy_device( T bump, int m, int n, T *A, int lda,
                        int gM, int m0, int n0, unsigned long long int seed ) {
  static constexpr int nbelem = detail::num_elems_v<T>;
  auto layout = Kokkos::LayoutStride(m, lda, n, 1);
  auto view = Kokkos::View<T**, Kokkos::LayoutStride,
                           KOKKOS_SPACE>(A, layout);

  //std::cout << "CORE_plgsy_device stream " << ttg::device::current_stream()
  //          << " buffer " << A
  //          << " m " << m << " n " << n
  //          << " lda " << lda << " gM " << gM << " m0 " << m0
  //          << " n0 " << n0 << " seed " << seed << std::endl;

  auto rnd = KOKKOS_LAMBDA(unsigned long long int n, unsigned long long int seed) {
    unsigned long long int a_k, c_k, ran;
    a_k = Rnd64_A;
    c_k = Rnd64_C;
    ran = seed;
    for (int i = 0; n; n >>= 1, ++i) {
      if (n & 1)
        ran = a_k * ran + c_k;
      c_k *= (a_k + 1);
      a_k *= a_k;
    }

    return ran;
  };

  auto gen = KOKKOS_LAMBDA(unsigned long long ran){
    if constexpr(detail::is_complex_v<T>) {
      return T((0.5f - ran * RndF_Mul), (0.5f - ran * RndF_Mul));
    } else {
      return (0.5f - ran * RndF_Mul);
    }
  };


  KOKKOS_POLICY pol = KOKKOS_POLICY_INSTANCE;

  if ( m0 == n0 ) {
    /* diagonal */
    Kokkos::parallel_for("diagonal",
      Kokkos::MDRangePolicy<KOKKOS_POLICY, Kokkos::Rank<2>>(pol, {0, 0}, {n, m}),
      KOKKOS_LAMBDA(int row, int col) {
        unsigned long long int jump = (unsigned long long int)m0 + (unsigned long long int)n0 * (unsigned long long int)gM;
        jump += std::min(col, row)*(gM+1);
        unsigned long long int ran;
        ran = rnd( nbelem * jump, seed );
        for (int i = 0; i < row; ++i) {
          for (int j = i; j < col; ++j) {
            ran = Rnd64_A*ran + Rnd64_C;
          }
        }
        view(row, col) = gen(ran);
        if (row == col) {
          /* bump diagonal */
          view(row, col) += bump;
        }
      }
    );
  } else if (m0 > n0) {
    /* Lower part */
    Kokkos::parallel_for("lower part",
      Kokkos::MDRangePolicy<KOKKOS_POLICY, Kokkos::Rank<2>>(pol, {0, 0}, {n, m}),
      KOKKOS_LAMBDA(int row, int col) {
        unsigned long long int jump = (unsigned long long int)m0 + (unsigned long long int)n0 * (unsigned long long int)gM;
        jump += row*gM;
        unsigned long long int ran;
        ran = rnd( nbelem * jump, seed );
        for (int i = 0; i < row*n+col; ++i) {
            ran = Rnd64_A*ran + Rnd64_C;
        }
        view(row, col) = gen(ran);
      }
    );
  } else {
    /* upper part */
    Kokkos::parallel_for("upper part",
      Kokkos::MDRangePolicy<KOKKOS_POLICY, Kokkos::Rank<2>>(pol, {0, 0}, {n, m}),
      KOKKOS_LAMBDA(int row, int col) {
        unsigned long long int jump = (unsigned long long int)m0 + (unsigned long long int)n0 * (unsigned long long int)gM;
        jump += col*gM;
        unsigned long long int ran;
        ran = rnd( nbelem * jump, seed );
        for (int i = 0; i < col*n+row; ++i) {
            ran = Rnd64_A*ran + Rnd64_C;
        }
        view(row, col) = gen(ran);
      }
    );
  }
}

/* implicit instantiations */
template void CORE_plgsy_device<float>(float bump, int m, int n, float *A, int lda,
                        int gM, int m0, int n0, unsigned long long int seed);

template void CORE_plgsy_device<double>(double bump, int m, int n, double *A, int lda,
                        int gM, int m0, int n0, unsigned long long int seed);

template void CORE_plgsy_device<std::complex<float>>(std::complex<float> bump, int m, int n, std::complex<float> *A, int lda,
                        int gM, int m0, int n0, unsigned long long int seed);

template void CORE_plgsy_device<std::complex<double>>(std::complex<double> bump, int m, int n, std::complex<double> *A, int lda,
                        int gM, int m0, int n0, unsigned long long int seed);


void kokkos_init(int& argc, char* argv[]) {
  Kokkos::initialize(argc, argv);
}

void kokkos_finalize() {
  Kokkos::finalize();
}