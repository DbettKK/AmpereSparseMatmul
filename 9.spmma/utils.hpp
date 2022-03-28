#include<algorithm>
#include<iostream>

#include<cuda_fp16.h>
#include<cuda_runtime_api.h> // cudaMalloc, cudaMemcpy, etc.
#include<cusparseLt.h>       // cusparseLt header

using namespace std;

using spmmaStatus_t = int;

// CUDA: grid stride looping
#define CUDA_KERNEL_LOOP(i, n)  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n);  i += blockDim.x * gridDim.x)

// without ret
#define CUDA_CHECK(func)                                                       \
{                                                                              \
    cudaError_t status = (func);                                               \
    if (status != cudaSuccess) {                                               \
        printf("CUDA API failed at line %d with error: %s (%d)\n",             \
               __LINE__, cudaGetErrorString(status), status);                  \
    }                                                                          \
}

#define CHECK_CUDA(func)                                                       \
{                                                                              \
    cudaError_t status = (func);                                               \
    if (status != cudaSuccess) {                                               \
        printf("CUDA API failed at line %d with error: %s (%d)\n",             \
               __LINE__, cudaGetErrorString(status), status);                  \
        return nullptr;                                                   \
    }                                                                          \
}

#define CHECK_CUSPARSE(func)                                                   \
{                                                                              \
    cusparseStatus_t status = (func);                                          \
    if (status != CUSPARSE_STATUS_SUCCESS) {                                   \
        printf("CUSPARSE API failed at line %d with error: %s (%d)\n",         \
               __LINE__, cusparseGetErrorString(status), status);              \
        return nullptr;                                                   \
    }                                                                          \
}

#define CUDA_POST_KERNEL_CHECK CUDA_CHECK(cudaPeekAtLastError())

// CUDA: use 512 threads per block
const int CUDA_NUM_THREADS = 512;

// CUDA: number of blocks for threads.
inline int GET_BLOCKS(const int N) {
  return (N + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS;
}

const spmmaStatus_t SUCCESS = 0;
const spmmaStatus_t DO_NOTHING = 1;
const spmmaStatus_t ERROR = 2;
const spmmaStatus_t UNSUPPORTED = 3;

template <typename Dtype>
void decimal2binary(Dtype num, int byteNum) {
    int *bottle = new int[byteNum];
    for (int i = 0; i < byteNum; i++) {
        bottle[i] = num & 1;
        num = num >> 1;

    }
    for (int i = byteNum - 1; i >= 0; i--) {
        if ((i + 1) % 4 == 0) printf(" ");
        printf("%d", bottle[i]);
    }
    delete[] bottle;
}

short convertIdx2Binary(int *index, int len) {
    short ret = 0;
    for (int i = len - 1; i >= 0; i--) {
        int item = index[i];
        ret = ret << 2;
        ret |= item;
    }
    return ret;
}

size_t get_cmpr_size(int row, int col) {
    int row_blocks = row % 32 ? row / 32 + 1 : row / 32;
    int col_blocks = col % 32 ? col / 32 + 1 : col / 32;
    return row_blocks * col_blocks * 32 * 32 / 8 > 256 ? row_blocks * col_blocks * 32 * 32 / 8 : 256;
}
