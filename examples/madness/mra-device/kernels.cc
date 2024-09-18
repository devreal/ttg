#include "ttg.h"
#include "util.h"

/* reuse dim3 from CUDA/HIP if available*/
#if !defined(TTG_HAVE_CUDA) && !defined(TTG_HAVE_HIP)
struct dim3 {
    int x, y, z;
};
#endif

/* define our own thread layout (single thread) */
static constexpr const dim3 threadIdx = {0, 0, 0};
static constexpr const dim3 blockIdx  = {0, 0, 0};
static constexpr const dim3 blockDim  = {1, 1, 1};
static constexpr const dim3 gridDim   = {1, 1, 1};

/* point cudaStream_t to our own stream type */
typedef ttg::device::Stream cudaStream_t;

/* include the CUDA code and hope that all is well */
#include "kernels.cu"