#include<cstdio>
#include<cstdlib>
#include<cstdint>
#include<cstring>

#include<fstream>
#include<iostream>
#include<vector>

#include<cublas_v2.h>
#include<cuda_runtime.h>
#include<cuda_runtime_api.h> // cudaMalloc, cudaMemcpy, etc.

#include "cuda_utils.h"

using namespace std;

using data_type = float;

int main(int argc, char *argv[]) {
    cublasHandle_t cublasH = NULL;
    cudaStream_t stream = NULL;

    const int m = m_global;
    const int n = n_global;
    const int k = k_global;
    const int lda = m; // 因为是列存储 因此ld代表行数
    const int ldb = k;
    const int ldc = m;

    float **matrices = read_bin(m, n, k);

    // 因为cublas的存储和普通的存在差别 因此需要进行一次倒换
    vector<data_type> A;
    int tmpA[k][m] = {};
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < k; j++) {
            tmpA[j][i] = matrices[0][i * k + j];
        }
    }
    for (int i = 0; i < k; i++) {
        for (int j = 0; j < m; j++) {
            A.push_back(tmpA[i][j]);
        }
    }

    vector<data_type> B;
    int tmpB[n][k] = {};
    for (int i = 0; i < k; i++) {
        for (int j = 0; j < n; j++) {
            tmpB[j][i] = matrices[1][i * n + j];
        }
    }
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < k; j++) {
            B.push_back(tmpB[i][j]);
        }
    }

    vector<data_type> C(m * n);
    const data_type alpha = 1.0;
    const data_type beta = 0.0;

    data_type *d_A = nullptr;
    data_type *d_B = nullptr;
    data_type *d_C = nullptr;

    cublasOperation_t transa = CUBLAS_OP_N;
    cublasOperation_t transb = CUBLAS_OP_N;

    printf("A\n");
    print_matrix(m, k, A.data(), lda);
    printf("=====\n");

    printf("B\n");
    print_matrix(k, n, B.data(), ldb);
    printf("=====\n");

    /* step 1: create cublas handle, bind a stream */
    CHECK_CUBLAS( cublasCreate(&cublasH) );

    CHECK_CUDA( cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) );
    CHECK_CUBLAS( cublasSetStream(cublasH, stream) );

    /* step 2: copy data to device */
    CHECK_CUDA( cudaMalloc(reinterpret_cast<void **>(&d_A), sizeof(data_type) * A.size()) );
    CHECK_CUDA( cudaMalloc(reinterpret_cast<void **>(&d_B), sizeof(data_type) * B.size()) );
    CHECK_CUDA( cudaMalloc(reinterpret_cast<void **>(&d_C), sizeof(data_type) * C.size()) );

    CHECK_CUDA( cudaMemcpyAsync(d_A, A.data(), sizeof(data_type) * A.size(), cudaMemcpyHostToDevice, stream) );
    CHECK_CUDA( cudaMemcpyAsync(d_B, B.data(), sizeof(data_type) * B.size(), cudaMemcpyHostToDevice, stream) );

    /* step 3: compute */
    CHECK_CUBLAS( cublasSgemm(cublasH, transa, transb, m, n, k, &alpha, d_A, lda, d_B, ldb, &beta, d_C, ldc) );

    /* step 4: copy data to host */
    CHECK_CUDA( cudaMemcpyAsync(C.data(), d_C, sizeof(data_type) * C.size(), cudaMemcpyDeviceToHost, stream));

    CHECK_CUDA( cudaStreamSynchronize(stream) );

    printf("C\n");
    print_matrix(m, n, C.data(), ldc);
    printf("=====\n");

    /* free resources */
    CHECK_CUDA( cudaFree(d_A) );
    CHECK_CUDA( cudaFree(d_B) );
    CHECK_CUDA( cudaFree(d_C) );

    CHECK_CUBLAS( cublasDestroy(cublasH) );

    CHECK_CUDA( cudaStreamDestroy(stream) );

    CHECK_CUDA( cudaDeviceReset() );

    return EXIT_SUCCESS;
}