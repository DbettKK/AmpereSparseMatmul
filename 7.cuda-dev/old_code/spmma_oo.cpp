#include<iostream>
#include<fstream>

#include<cstdint>
#include<cstdio>
#include<cstring>       // memset
#include<cstdlib>       // malloc
#include<cmath>

#include<cuda_fp16.h>
#include<cuda_runtime_api.h> // cudaMalloc, cudaMemcpy, etc.
#include<cusparseLt.h>       // cusparseLt header

#include"spmma_oo.hpp"

using namespace std;

using spmmaStatus_t = int;

spmmaStatus_t check_gpu() {
    // 检查GPU是否支持cuSparseLt
    int major_cc, minor_cc;
    CHECK_CUDA( cudaDeviceGetAttribute(&major_cc, cudaDevAttrComputeCapabilityMajor, 0) )
    CHECK_CUDA( cudaDeviceGetAttribute(&minor_cc, cudaDevAttrComputeCapabilityMinor, 0) )
    if (!(major_cc == 8 && minor_cc == 0) && !(major_cc == 8 && minor_cc == 6)) {
        printf("\n cusparseLt is supported only on GPU devices with compute capability == 8.0, 8.6 current: %d.%d\n\n",
                     major_cc, minor_cc);
        return UNSUPPORTED;
    }
}

spmmaStatus_t __mma_matmul(MatrixParam *param, bool isValid) {
    __half *hA = param->A;
    __half *hB = param->B;
    __half *hC = param->C;

    int m = param->m, k = param->k, n = param->n;

    size_t A_size = m * k * sizeof(__half);
    size_t B_size = k * n * sizeof(__half);
    size_t C_size = m * n * sizeof(__half);

    // Leading dimension 如果行优先则代表列数
    int lda = k, ldb = n, ldc = n;
    auto opA = CUSPARSE_OPERATION_NON_TRANSPOSE;
    auto opB = CUSPARSE_OPERATION_NON_TRANSPOSE;
    auto order = CUSPARSE_ORDER_ROW; // cusparseOrder_t
    auto type  = CUDA_R_16F;
    auto compute_type = CUSPARSE_COMPUTE_16F;
    float alpha = 1.0f;
    float beta  = 0.0f;
    unsigned alignment = 16;

    // device
    __half *dA, *dB, *dC, *dD, *dB_compressed;
    int *d_valid;
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
    CHECK_CUSPARSE( cusparseLtStructuredDescriptorInit(&handle, &matB, k, n, ldb, alignment, type, order, CUSPARSELT_SPARSITY_50_PERCENT) )
    CHECK_CUSPARSE( cusparseLtDenseDescriptorInit(&handle, &matA, m, k, lda, alignment, type, order) )
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
    // Prune and Compress
    if (!isValid) {
        // 不符合条件 需要进行压缩
        CHECK_CUSPARSE( cusparseLtSpMMAPruneCheck(&handle, &matmul, dB, d_valid, stream) )
        int is_valid;
        CHECK_CUDA( cudaMemcpyAsync(&is_valid, d_valid, sizeof(d_valid), cudaMemcpyDeviceToHost, stream) )
        CHECK_CUDA( cudaStreamSynchronize(stream) )
        if (is_valid != 0) {
            std::printf("!!!! The matrix need to be pruned.\n");
            CHECK_CUSPARSE( cusparseLtSpMMAPrune(&handle, &matmul, dB, dB, CUSPARSELT_PRUNE_SPMMA_TILE, stream) )
        }
        // 需要把prune后的b拿出来 和cpu比较需要用
//        __half *newB = new __half[k * n];
//        CHECK_CUDA( cudaMemcpy(newB, dB, B_size, cudaMemcpyDeviceToHost) )
//        param->B = newB;
    }
    // 符合条件 不用判断 直接compress即可
    CHECK_CUSPARSE( cusparseLtSpMMACompressedSize(&handle, &plan, &compressed_size) )
    CHECK_CUDA( cudaMalloc((void**) &dB_compressed, compressed_size) )
    CHECK_CUSPARSE( cusparseLtSpMMACompress(&handle, &plan, dB, dB_compressed, stream) )
    //--------------------------------------------------------------------------

    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // Search the best kernel
    void* d_workspace = nullptr;
    int num_streams = 0;
    cudaStream_t* streams = nullptr;
    /*
    int alg_id;
    CHECK_CUSPARSE( cusparseLtMatmulAlgGetAttribute(&handle, &alg_sel, CUSPARSELT_MATMUL_ALG_CONFIG_ID, &alg_id, sizeof(alg_id)) )
    printf("best alg: %d\n", alg_id);
    */
    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // Perform the matrix multiplication
    CHECK_CUSPARSE( cusparseLtMatmul(&handle, &plan, &alpha, dA, dB_compressed, &beta, dC, dD, d_workspace, streams, num_streams) )
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
    CHECK_CUDA( cudaMemcpy(param->D, dD, C_size, cudaMemcpyDeviceToHost) )

    return SUCCESS;
}

spmmaStatus_t __mma_matmul_A(MatrixParam *param, __half *matA_cmpr) {
    __half *hA = param->A;
    __half *hB = param->B;
    __half *hC = param->C;

    int m = param->m, k = param->k, n = param->n;

    size_t A_size = m * k * sizeof(__half);
    size_t B_size = k * n * sizeof(__half);
    size_t C_size = m * n * sizeof(__half);

    // Leading dimension 如果行优先则代表列数
    int lda = k, ldb = n, ldc = n;
    auto opA = CUSPARSE_OPERATION_NON_TRANSPOSE;
    auto opB = CUSPARSE_OPERATION_NON_TRANSPOSE;
    auto order = CUSPARSE_ORDER_ROW; // cusparseOrder_t
    auto type  = CUDA_R_16F;
    auto compute_type = CUSPARSE_COMPUTE_16F;
    float alpha = 1.0f;
    float beta  = 0.0f;
    unsigned alignment = 16;

    // device
    __half *dA, *dB, *dC, *dD, *dA_compressed;
    int *d_valid;
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
    CHECK_CUSPARSE( cusparseLtStructuredDescriptorInit(&handle, &matA, m, k, lda, alignment, type, order, CUSPARSELT_SPARSITY_50_PERCENT) )
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

    // Prune and Compress

    cout << "A: " << endl;
    param->print_matrix(param->A, m, k);
    cout << "A_cmpr: " << endl;
    param->print_matrix(matA_cmpr, m, k / 2);

    CHECK_CUSPARSE( cusparseLtSpMMACompressedSize(&handle, &plan, &compressed_size) )
    CHECK_CUDA( cudaMalloc((void**) &dA_compressed, compressed_size) )
    cout << compressed_size / sizeof(void) << endl;
    CHECK_CUSPARSE( cusparseLtSpMMACompress(&handle, &plan, dA, dA_compressed, stream) )
    //--------------------------------------------------------------------------

    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // Search the best kernel
    void* d_workspace = nullptr;
    int num_streams = 0;
    cudaStream_t* streams = nullptr;
    /*
    int alg_id;
    CHECK_CUSPARSE( cusparseLtMatmulAlgGetAttribute(&handle, &alg_sel, CUSPARSELT_MATMUL_ALG_CONFIG_ID, &alg_id, sizeof(alg_id)) )
    printf("best alg: %d\n", alg_id);
    */
    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // Perform the matrix multiplication
    CHECK_CUSPARSE( cusparseLtMatmul(&handle, &plan, &alpha, dA_compressed, dB, &beta, dC, dD, d_workspace, streams, num_streams) )
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
    CHECK_CUDA( cudaMemcpy(param->D, dD, C_size, cudaMemcpyDeviceToHost) )

    return SUCCESS;
}

/* matmul with mma */
/* matD -> OUT, matC/matA_cmpr -> alternative */
MatrixParam* spmma_matmul(MatrixParam *param, bool isMatrixValid) {
    // 1. fix matrix
    if (param->C == nullptr) {
        param->C = new __half[param->m * param->n];
        memset(param->C, 0, param->m * param->n * sizeof(__half));
    }
    if (param->D == nullptr) {
        param->D = new __half[param->m * param->n];
        memset(param->D, 0, param->m * param->n * sizeof(__half));
    }

    MatrixParam *out = param->fix_matrix();

    // 2.calculate
    __mma_matmul(out, isMatrixValid);

    // 3. compare with cpu
    //out->check_correct();

    return out;
}

Tensor4d *spmma_conv(ConvParam *param) {
    MatrixParam *matrix = param->im2col();  // 最初版本的matrix
    MatrixParam *ans = spmma_matmul(matrix, false);   // 这是fix后并且计算了D的matrix
    MatrixParam *refix = ans->refix_matrix(matrix->m, matrix->n);    // 是把D重新恢复的matrix 其他都不变
    //refix->print_all();
    Tensor4d *ret = param->im2col_rev(refix);
    return ret;
}
// 5866
void test_gemm(int m, int k, int n) {
    MatrixParam *param = new MatrixParam(m, k, n);
    __half *cmpr = param->generate_sparse_cmpr(5);
    MatrixParam *ans = spmma_matmul(param, false);
    //ans->check_correct();
    // compress b的时候 是反过来的
    //ans->check_correct();
}

void test_conv() {
    Tensor4d *data = new Tensor4d(4, 3, 256, 256);
    Tensor4d *kernel = new Tensor4d(64, 3, 7, 7);
    data->read_bin("data.bin");
    kernel->read_bin("kernel.bin");

    Tensor4d *ans = spmma_conv(new ConvParam(data, kernel, 0, 1));
    //ans->print_tensor();
}

int main() {
    test_conv();
}

// todo: cpu时间的考虑
// todo: A B sparse的多次比较 以及转置的考虑
// todo: im2col和padding放到gpu做
// todo: 和tvm比较时间