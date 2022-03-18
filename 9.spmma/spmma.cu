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

#include"spmma.hpp"

using namespace std;

/*
    matA, matB: device IN
*/
MatrixParam *spmma_matmul(const __half *matA_h, const __half *matB_h, int m_old, int k_old, int n_old, bool isValid) {
    int m = m_old % 8 ? m_old + 8 - m_old % 8 : m_old;
    int k = k_old % 16 ? k_old + 16 - k_old % 16 : k_old;
    int n = n_old % 8 ? n_old + 8 - n_old % 8 : n_old;

    MatrixParam *ret = new MatrixParam(m, k, n);

    size_t A_size = m * k * sizeof(__half);
    size_t B_size = k * n * sizeof(__half);
    size_t C_size = m * n * sizeof(__half);
    // device
    __half *dA, *dB, *dC, *dD, *dA_compressed;
    int *d_valid;
    CHECK_CUDA( cudaMalloc((void**) &dA, A_size) )
    CHECK_CUDA( cudaMalloc((void**) &dB, B_size) )
    CHECK_CUDA( cudaMalloc((void**) &dC, C_size) )
    CHECK_CUDA( cudaMalloc((void**) &d_valid, sizeof(d_valid)) )
    dD = dC;

    __padding_matrix<__half>(matA_h, m_old, k_old, dA, m, k);
    __padding_matrix<__half>(matB_h, k_old, n_old, dB, k, n);
    CHECK_CUDA( cudaMemset(dC, 0, C_size) )

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
    if (!isValid) {
        // 不符合条件 需要进行压缩
        CHECK_CUSPARSE( cusparseLtSpMMAPruneCheck(&handle, &matmul, dA, d_valid, stream) )
        int is_valid;
        CHECK_CUDA( cudaMemcpyAsync(&is_valid, d_valid, sizeof(d_valid), cudaMemcpyDeviceToHost, stream) )
        CHECK_CUDA( cudaStreamSynchronize(stream) )
        if (is_valid != 0) {
            std::printf("!!!! The matrix need to be pruned.\n");
            CHECK_CUSPARSE( cusparseLtSpMMAPrune(&handle, &matmul, dA, dA, CUSPARSELT_PRUNE_SPMMA_TILE, stream) )
        }
    }
    // 符合条件 不用判断 直接compress即可
    CHECK_CUSPARSE( cusparseLtSpMMACompressedSize(&handle, &plan, &compressed_size) )
    CHECK_CUDA( cudaMalloc((void**) &dA_compressed, compressed_size) )
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

    ret->A = new __half[m_old * k_old];
    ret->B = new __half[k_old * n_old];
    ret->C = new __half[m_old * n_old];
    ret->D = new __half[m_old * n_old];
    CHECK_CUDA( cudaMemcpy(ret->A, dA, m_old * k_old * sizeof(__half), cudaMemcpyDeviceToHost) )
    CHECK_CUDA( cudaMemcpy(ret->B, dB, k_old * n_old * sizeof(__half), cudaMemcpyDeviceToHost) )
    memset(ret->C, 0, m_old * n_old);
    CHECK_CUDA( cudaMemcpy2D(ret->D, n_old * sizeof(__half), dD, n * sizeof(__half), n_old * sizeof(__half), m_old, cudaMemcpyDeviceToHost) )

    return ret;
}

Tensor4d *spmma_conv(ConvParam *param) {
    __half *d_data, *d_kernel, *d_im2col;
    CUDA_CHECK( cudaMalloc((void **)&d_data, param->data->get_size() * sizeof(__half)) )
    CUDA_CHECK( cudaMemcpy(d_data, param->data->tensor, param->data->get_size() * sizeof(__half), cudaMemcpyHostToDevice) )
    CUDA_CHECK( cudaMalloc((void **)&d_kernel, param->kernel->get_size() * sizeof(__half)) )
    CUDA_CHECK( cudaMemcpy(d_kernel, param->kernel->tensor, param->kernel->get_size() * sizeof(__half), cudaMemcpyHostToDevice) )
    CUDA_CHECK( cudaMalloc((void **)&d_im2col, param->getIm2col_size() * sizeof(__half)) )

    im2col_gpu<__half>(d_data, param->data->n, param->data->c, param->data->h, param->data->w,
        param->kernel->h, param->kernel->w, param->padding, param->padding, param->stride, param->stride, 1, 1, d_im2col);

    __half *tmp_d = new __half[param->getIm2col_size()];
    cudaMemcpy(tmp_d, d_im2col, param->getIm2col_size() * sizeof(__half), cudaMemcpyDeviceToHost);
    printf("im2col: \n");
    for (int i = 0; i < param->kernel->c * param->kernel->h * param->kernel->w; i++) {
        for (int j = 0; j < param->data->n * param->getOut_height() * param->getOut_width(); j++){
            printf("%d ", __half2int_rz(tmp_d[i * param->data->n * param->getOut_height() * param->getOut_width() + j]));
        }
        printf("\n");
    }

    MatrixParam *out = spmma_matmul(d_kernel, d_im2col, param->kernel->n, param->kernel->c * param->kernel->h * param->kernel->w,
        param->data->n * param->getOut_height() * param->getOut_width(), false);

    //refix->print_all();
    __half *d_ans, *d_im2col_rev;
    int im2col_size = param->data->n * param->kernel->n * param->getOut_height() * param->getOut_width();
    CUDA_CHECK( cudaMalloc((void **)&d_ans, out->m * out->n * sizeof(__half)) )
    CUDA_CHECK( cudaMemcpy(d_ans, out->D, out->m * out->n * sizeof(__half), cudaMemcpyHostToDevice) )
    CUDA_CHECK( cudaMalloc((void **)&d_im2col_rev, im2col_size * sizeof(__half)) )
    int num_kernels = param->data->n * param->getOut_height() * param->getOut_width();
    im2col_rev_kernel<<<GET_BLOCKS(num_kernels), CUDA_NUM_THREADS>>>(
        num_kernels, d_ans, param->data->n, param->kernel->n, param->getOut_height(), param->getOut_width(), d_im2col_rev);

    __half *im2col_rev = new __half[im2col_size];
    CUDA_CHECK( cudaMemcpy(im2col_rev, d_im2col_rev, im2col_size * sizeof(__half), cudaMemcpyDeviceToHost) )
    return new Tensor4d(im2col_rev, param->data->n, param->kernel->n, param->getOut_height(), param->getOut_width());
}

void test_matmul() {
    int m = 16, k = 16, n = 16;
    MatrixParam *param = new MatrixParam(m, k, n);
    param->read_bin("a.bin", "b.bin", "c.bin");
    __half *dA, *dB;
    cudaMalloc((void **)&dA, m * k * sizeof(__half));
    cudaMalloc((void **)&dB, k * n * sizeof(__half));
    cudaMemcpy(dA, param->A, m * k * sizeof(__half), cudaMemcpyHostToDevice);
    cudaMemcpy(dB, param->B, k * n * sizeof(__half), cudaMemcpyHostToDevice);

    MatrixParam *out = spmma_matmul(dA, dB, m, k, n, false);

    out->check_correct();
}

void test_conv() {
    int data_n = 4, data_c = 3, data_h = 16, data_w = 16;
    int kernel_n = 2, kernel_c = 3, kernel_h = 3, kernel_w = 3;
    Tensor4d *data = new Tensor4d(data_n, data_c, data_h, data_w);
    Tensor4d *kernel = new Tensor4d(kernel_n, kernel_c, kernel_h, kernel_w);
    data->read_bin("data.bin");
    kernel->read_bin("kernel.bin");
    printf("data:\n");
    data->print_tensor();
    ConvParam *param = new ConvParam(data, kernel, 0, 1);
    Tensor4d *out = spmma_conv(param);
    out->print_tensor();
}


int main() {
    test_conv();
}
