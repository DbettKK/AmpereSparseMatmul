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

#include "cublas_utils.h"

using namespace std;

using data_type = double;


double **read_bin(int m, int n, int k) {
    double **ret = (double **)malloc(sizeof(double *) * 3);
    __half *mat_a_host = new __half[m * k];
    __half *mat_b_host = new __half[k * n];
    __half *mat_c_host = new __half[m * n];
    ifstream a_fs("a.bin", ios_base::binary);
    a_fs.read((char *)mat_a_host, m * k * sizeof(__half));
    ifstream b_fs("b.bin", ios_base::binary);
    b_fs.read((char *)mat_b_host, k * n * sizeof(__half));
    ifstream c_fs("c.bin", ios_base::binary);
    c_fs.read((char *)mat_c_host, m * n * sizeof(__half));

    double *a_float = new double[m * k];
    double *b_float = new double[k * n];
    double *c_float = new double[m * n];
    for (int i = 0; i < m * k; i++) {
        a_float[i] = static_cast<double>(mat_a_host[i]);
    }
    for (int i = 0; i < n * k; i++) {
        b_float[i] = static_cast<double>(mat_b_host[i]);
    }
    for (int i = 0; i < m * n; i++) {
        c_float[i] = static_cast<double>(mat_c_host[i]);
    }
    ret[0] = a_float;
    ret[1] = b_float;
    ret[2] = c_float;
    return ret;
}


int main(int argc, char *argv[]) {
    cublasHandle_t cublasH = NULL;
    cudaStream_t stream = NULL;

    const int m = 16;
    const int n = 16;
    const int k = 8;
    const int lda = m;
    const int ldb = k;
    const int ldc = m;

    double **matrices = read_bin(m, n, k);



    /*
     *   A = | 1.0 | 2.0 |
     *       | 3.0 | 4.0 |
     *
     *   B = | 5.0 | 6.0 |
     *       | 7.0 | 8.0 |
     */

    vector<data_type> A;
    for (int i = 0; i < m * k; i++) {
        A.push_back(matrices[0][i]);
    }
    vector<data_type> B;
    for (int i = 0; i < n * k; i++) {
        B.push_back(matrices[1][i]);
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
    CUBLAS_CHECK(cublasCreate(&cublasH));

    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    CUBLAS_CHECK(cublasSetStream(cublasH, stream));

    /* step 2: copy data to device */
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_A), sizeof(data_type) * A.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_B), sizeof(data_type) * B.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_C), sizeof(data_type) * C.size()));

    CUDA_CHECK(cudaMemcpyAsync(d_A, A.data(), sizeof(data_type) * A.size(), cudaMemcpyHostToDevice,
                               stream));
    CUDA_CHECK(cudaMemcpyAsync(d_B, B.data(), sizeof(data_type) * B.size(), cudaMemcpyHostToDevice,
                               stream));

    /* step 3: compute */
    CUBLAS_CHECK(
        cublasDgemm(cublasH, transa, transb, m, n, k, &alpha, d_A, lda, d_B, ldb, &beta, d_C, ldc));

    /* step 4: copy data to host */
    CUDA_CHECK(cudaMemcpyAsync(C.data(), d_C, sizeof(data_type) * C.size(), cudaMemcpyDeviceToHost,
                               stream));

    CUDA_CHECK(cudaStreamSynchronize(stream));

    /*
     *   C = | 23.0 | 31.0 |
     *       | 34.0 | 46.0 |
     */

    printf("C\n");
    print_matrix(m, n, C.data(), ldc);
    printf("=====\n");

    /* free resources */
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    CUBLAS_CHECK(cublasDestroy(cublasH));

    CUDA_CHECK(cudaStreamDestroy(stream));

    CUDA_CHECK(cudaDeviceReset());

    return EXIT_SUCCESS;
}