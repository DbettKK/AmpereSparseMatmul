#include <cuda_runtime_api.h> // cudaMalloc, cudaMemcpy, etc.
#include <cusparseLt.h>       // cusparseLt header
#include<cstdint>
#include<cstdio>
#include<cstring>
#include<cuda_fp16.h>
#include<fstream>
#include<iostream>

using namespace std;

// spmma: 16x16 16x8 -> 16x8 m16n8k16

#define CHECK_CUDA(func)                                                       \
{                                                                              \
    cudaError_t status = (func);                                               \
    if (status != cudaSuccess) {                                               \
        printf("CUDA API failed at line %d with error: %s (%d)\n",             \
               __LINE__, cudaGetErrorString(status), status);                  \
        return EXIT_FAILURE;                                                   \
    }                                                                          \
}

// 记录矩阵的size是否发生变化 全局变量
int m_fix = 0, k_fix = 0, n_fix = 0;

void init() {
    // 检查GPU是否支持cuSparseLt
    int major_cc, minor_cc;
    CHECK_CUDA( cudaDeviceGetAttribute(&major_cc, cudaDevAttrComputeCapabilityMajor, 0) )
    CHECK_CUDA( cudaDeviceGetAttribute(&minor_cc, cudaDevAttrComputeCapabilityMinor, 0) )
    if (!(major_cc == 8 && minor_cc == 0) && !(major_cc == 8 && minor_cc == 6)) {
        printf("\n cusparseLt is supported only on GPU devices with compute capability == 8.0, 8.6 current: %d.%d\n\n",
                     major_cc, minor_cc);
        return EXIT_UNSUPPORTED;
    }
}

void input(__half *hA, __half *hB, __half *hC, int m, int n, int k) {
    //__half hA[m * k];
    //__half hB[k * n];
    //__half hC[m * n] = {};
    // 必须要求hA hB hC不是常量指针
    hA = handle_input(hA, m, k, 0);
    hB = handle_input(hB, k, n, 1);
    hC = handle_input(hC, m, n, 2);

    int A_size = m * k * sizeof(__half);
    int B_size = k * n * sizeof(__half);
    int C_size = m * n * sizeof(__half);

    __half *dA, *dB, *dC, *dD, *dA_compressed;
    int    *d_valid;
    CHECK_CUDA( cudaMalloc((void**) &dA, A_size) )
    CHECK_CUDA( cudaMalloc((void**) &dB, B_size) )
    CHECK_CUDA( cudaMalloc((void**) &dC, C_size) )
    CHECK_CUDA( cudaMalloc((void**) &d_valid, sizeof(d_valid)) )
    dD = dC;

    CHECK_CUDA( cudaMemcpy(dA, hA, A_size, cudaMemcpyHostToDevice) )
    CHECK_CUDA( cudaMemcpy(dB, hB, B_size, cudaMemcpyHostToDevice) )
    CHECK_CUDA( cudaMemcpy(dC, hC, C_size, cudaMemcpyHostToDevice) )
}

__half *handle_input(__half *item, int m, int n, int flag) {
    if (m % 8 == 0 && n % 8 == 0) {
        return item;
    }
    if (m % 8 == 0) {
        int fix = 8 - n % 8;
        __half *ret = (__half *)malloc(m * (n + fix) * sizeof(__half));
        int ret_cnt = 0;
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                ret[ret_cnt++] = item[i * n + j];
            }
            for (int j = 0; j < fix; j++) {
                ret[ret_cnt++] = 0;
            }
        }
        if (flag == 1 || flag = 2) {
            n_fix = fix;
        }
        if (flag == 0) {
            k_fix = fix;
        }
        return ret;
    }
    if (n % 8 == 0) {
        int fix = 8 - m % 8;
        __half *ret = (__half *)malloc((m + fix) * n * sizeof(__half));
        memset(ret, 0, (m + fix) * n * sizeof(__half));
        memcpy(ret, item, m * n * sizeof(__half));
        if (flag == 0 || flag == 2) {
            m_fix = fix;
        }
        if (flag == 1) {
            k_fix = fix;
        }
        return ret;
    }
    int fix_m = 8 - m % 8;
    int fix_n = 8 - n % 8;
    __half *ret = (__half *)malloc((m + fix_m) * (n + fix_n) * sizeof(__half));
    memset(ret, 0, (m + fix_m) * (n + fix_n) * sizeof(__half));
    int ret_cnt = 0;
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            ret[ret_cnt++] = item[i * n + j];
        }
        for (int j = 0; j < fix_n; j++) {
            ret[ret_cnt++] = 0;
        }
    }
    if (flag == 0) {
        m_fix = fix_m;
        k_fix = fix_n;
    }
    if (flag == 1) {
        n_fix = fix_n;
        k_fix = fix_m;
    }
    if (flag == 2) {
        m_fix = fix_m;
        n_fix = fix_n;
    }
    return ret;
}

__half *handle_output(__half *item, int m, int n) {
    if (!m_fix && !n_fix) {
        return item;
    }
    __half *ret = (__half *)malloc((m - m_fix) * (n - n_fix) * sizeof(__half));
    for (int i = 0; i < m - m_fix; i++) {
        for (int j = 0; j < n - n_fix; j++) {
            ret[i * (n - n_fix) + j] = item[i * n + j];
        }
    }
    return ret;
}

void tile(__half *item, int row, int col) {

}

void calculate(__half *hA, __half *hB, __half *hC,  __half *hD, int m, int n, int k) {
    int A_size = m * k * sizeof(__half);
    int B_size = k * n * sizeof(__half);
    int C_size = m * n * sizeof(__half);

    // Leading dimension 如果行优先则代表列数
    int lda = k, ldb = n, ldc = n;

    unsigned alignment = 16;

    auto order = CUSPARSE_ORDER_ROW; // cusparseOrder_t
    auto type  = CUDA_R_16F;
    auto compute_type = CUSPARSE_COMPUTE_16F;

    __half *dA, *dB, *dC, *dD, *dA_compressed;
    int    *d_valid;
    CHECK_CUDA( cudaMalloc((void**) &dA, A_size) )
    CHECK_CUDA( cudaMalloc((void**) &dB, B_size) )
    CHECK_CUDA( cudaMalloc((void**) &dC, C_size) )
    CHECK_CUDA( cudaMalloc((void**) &d_valid, sizeof(d_valid)) )
    dD = dC;
    // 从host拷贝到device
    CHECK_CUDA( cudaMemcpy(dA, hA, A_size, cudaMemcpyHostToDevice) )
    CHECK_CUDA( cudaMemcpy(dB, hB, B_size, cudaMemcpyHostToDevice) )
    CHECK_CUDA( cudaMemcpy(dC, hC, C_size, cudaMemcpyHostToDevice) )

    //--------------------------------------------------------------------------
    cusparseLtHandle_t             handle;
    cusparseLtMatDescriptor_t      matA, matB, matC;
    cusparseLtMatmulDescriptor_t   matmul;
    cusparseLtMatmulAlgSelection_t alg_sel;
    cusparseLtMatmulPlan_t         plan;
    cudaStream_t                   stream = nullptr;
    CHECK_CUSPARSE( cusparseLtInit(&handle) )
    // matrix descriptor initialization
    CHECK_CUSPARSE( cusparseLtStructuredDescriptorInitcusparseLtStructuredDescriptorInit(&handle, &matA, m, k, lda, alignment, type, order, CUSPARSELT_SPARSITY_50_PERCENT) )
    CHECK_CUSPARSE( cusparseLtDenseDescriptorInit(&handle, &matB, k, n, ldb, alignment, type, order) )
    CHECK_CUSPARSE( cusparseLtDenseDescriptorInit(&handle, &matC, m, n, ldc, alignment, type, order) )
    // matmul, algorithm selection, and plan initialization
    CHECK_CUSPARSE( cusparseLtMatmulDescriptorInit( &handle, &matmul, opA, opB, &matA, &matB, &matC, &matC, compute_type) )
    CHECK_CUSPARSE( cusparseLtMatmulAlgSelectionInit( &handle, &alg_sel, &matmul, CUSPARSELT_MATMUL_ALG_DEFAULT) )
    int alg = 0;    // 算法
    CHECK_CUSPARSE( cusparseLtMatmulAlgSetAttribute( &handle, &alg_sel, CUSPARSELT_MATMUL_ALG_CONFIG_ID, &alg, sizeof(alg)))
    size_t workspace_size, compressed_size;
    CHECK_CUSPARSE( cusparseLtMatmulGetWorkspace(&handle, &alg_sel, &workspace_size))

    CHECK_CUSPARSE( cusparseLtMatmulPlanInit(&handle, &plan, &matmul, &alg_sel, workspace_size) )
    //--------------------------------------------------------------------------
    // Prune the A matrix (in-place) and check the correcteness
    CHECK_CUSPARSE( cusparseLtSpMMAPrune(&handle, &matmul, dA, dA, CUSPARSELT_PRUNE_SPMMA_TILE, stream) )
    CHECK_CUSPARSE( cusparseLtSpMMAPruneCheck(&handle, &matmul, dA, d_valid, stream) )
    int is_valid;
    CHECK_CUDA( cudaMemcpyAsync(&is_valid, d_valid, sizeof(d_valid), cudaMemcpyDeviceToHost, stream) )
    CHECK_CUDA( cudaStreamSynchronize(stream) )
    if (is_valid != 0) {
        std::printf("!!!! The matrix has been pruned in a wrong way. "
                    "cusparseLtMatmul will not provide correct results\n");
        return EXIT_FAILURE;
    }
    //--------------------------------------------------------------------------
    // Compress the A matrix
    CHECK_CUSPARSE( cusparseLtSpMMACompressedSize(&handle, &plan, &compressed_size) )
    CHECK_CUDA( cudaMalloc((void**) &dA_compressed, compressed_size) )

    CHECK_CUSPARSE( cusparseLtSpMMACompress(&handle, &plan, dA, dA_compressed, stream) )
    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // Search the best kernel
    void*         d_workspace = nullptr;
    int           num_streams = 0;
    cudaStream_t* streams     = nullptr;
    CHECK_CUSPARSE( cusparseLtMatmulSearch(&handle, &plan, &alpha, dA_compressed, dB, &beta, dC,dD, d_workspace,
                                           streams, num_streams) )
    int alg_id;
    CHECK_CUSPARSE( cusparseLtMatmulAlgGetAttribute(&handle, &alg_sel, CUSPARSELT_MATMUL_ALG_CONFIG_ID,
                                           &alg_id, sizeof(alg_id)) )
    printf("best alg: %d\n", alg_id);
    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // Perform the matrix multiplication
    CHECK_CUSPARSE( cusparseLtMatmul(&handle, &plan, &alpha, dA_compressed, dB,
                                     &beta, dC, dD, d_workspace, streams,
                                     num_streams) )
    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // destroy plan and handle
    CHECK_CUSPARSE( cusparseLtMatDescriptorDestroy(&matA) )
    CHECK_CUSPARSE( cusparseLtMatDescriptorDestroy(&matB) )
    CHECK_CUSPARSE( cusparseLtMatDescriptorDestroy(&matC) )
    CHECK_CUSPARSE( cusparseLtMatmulPlanDestroy(&plan) )
    CHECK_CUSPARSE( cusparseLtDestroy(&handle) )
    //--------------------------------------------------------------------------
    // device result check
    // matrix A has been pruned
    CHECK_CUDA( cudaMemcpy(hA, dA, A_size, cudaMemcpyDeviceToHost) )
    CHECK_CUDA( cudaMemcpy(hC, dC, C_size, cudaMemcpyDeviceToHost) )

}

void print(__half *item, int row, int col) {
    for (int i = 0; i < row; i++) {
        for (int j = 0; j < col; j++) {
            cout << item[i * col + j] << " ";
        }
        cout << endl;
    }
}

void expose(__half *hA, __half *hB, __half *hC, int m, int n, int k) {
    init();
    hA = handle_input(hA, m, k, 0);
    hB = handle_input(hB, k, n, 1);
    hC = handle_input(hC, m, n, 2);
    m = m + m_fix;
    n = n + n_fix;
    k = k + k_fix;
    __half *hD = (__half *)malloc(m * n * sizeof(__half));
    calculate(hA, hB, hC, hD, m, n, k);
    __half *output = handle_output(hD, m, n);
    print(output, m, n);
}

int main() {
    int m = 0, k = 0, n = 0;
    __half *hA = (__half *)malloc(m * k * sizeof(__half));
    __half *hB = (__half *)malloc(k * n * sizeof(__half));
    __half *hC = (__half *)malloc(m * n * sizeof(__half));
    expose(hA, hB, hC, m, n, k);
}