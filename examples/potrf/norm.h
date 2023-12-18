#pragma once

#include "../devblas_helper.h"
#include "../matrixtile.h"


#if defined(TTG_HAVE_CUDART) || defined(TTG_HAVE_HIP)

static void device_norm(const MatrixTile<double> &A, double *norm) {
auto size = A.size();
auto buffer = A.buffer().current_device_ptr();
std::cout << "device_norm ptr " << buffer << " device " << ttg::device::current_device()
          << " stream " << ttg::device::current_stream() << std::endl;
#if defined(TTG_HAVE_CUDA)
auto handle = cublas_handle();
//double n = 1.0;
cublasDnrm2(handle, size, buffer, 1, norm);
#elif defined(TTG_HAVE_HIPBLAS)
hipblasDnrm2(hipblas_handle(), size, buffer, 1, norm);
#endif
}
#endif // defined(TTG_HAVE_CUDART) || defined(TTG_HAVE_HIP)