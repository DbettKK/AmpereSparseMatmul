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

//#include "cublas_utils.h"

using namespace std;

using data_type = float;

#define CUDA_CHECK(err)                                                                            \
    do {                                                                                           \
        cudaError_t err_ = (err);                                                                  \
        if (err_ != cudaSuccess) {                                                                 \
            std::printf("CUDA error %d at %s:%d\n", err_, __FILE__, __LINE__);                     \
            throw std::runtime_error("CUDA error");                                                \
        }                                                                                          \
    } while (0)

// cublas API error checking
#define CUBLAS_CHECK(err)                                                                          \
    do {                                                                                           \
        cublasStatus_t err_ = (err);                                                               \
        if (err_ != CUBLAS_STATUS_SUCCESS) {                                                       \
            std::printf("cublas error %d at %s:%d\n", err_, __FILE__, __LINE__);                   \
            throw std::runtime_error("cublas error");                                              \
        }                                                                                          \
    } while (0)


void print_matrix(const int &m, const int &n, const float *A, const int &lda) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            std::printf("%0.2f ", A[j * lda + i]);
        }
        std::printf("\n");
    }
}

data_type **read_bin(int m, int n, int k) {
    data_type **ret = (data_type **)malloc(sizeof(data_type *) * 3);
    data_type *mat_a_host = new data_type[m * k];
    data_type *mat_b_host = new data_type[k * n];
    data_type *mat_c_host = new data_type[m * n];
    ifstream a_fs("a.bin", ios_base::binary);
    a_fs.read((char *)mat_a_host, m * k * sizeof(data_type));
    ifstream b_fs("b.bin", ios_base::binary);
    b_fs.read((char *)mat_b_host, k * n * sizeof(data_type));
    ifstream c_fs("c.bin", ios_base::binary);
    c_fs.read((char *)mat_c_host, m * n * sizeof(data_type));

//    double *a_float = new double[m * k];
//    double *b_float = new double[k * n];
//    double *c_float = new double[m * n];
//    for (int i = 0; i < m * k; i++) {
//        a_float[i] = static_cast<double>(mat_a_host[i]);
//    }
//    for (int i = 0; i < n * k; i++) {
//        b_float[i] = static_cast<double>(mat_b_host[i]);
//    }
//    for (int i = 0; i < m * n; i++) {
//        c_float[i] = static_cast<double>(mat_c_host[i]);
//    }
    ret[0] = mat_a_host;
    ret[1] = mat_b_host;
    ret[2] = mat_c_host;
    return ret;
}


int main(int argc, char *argv[]) {
    cublasHandle_t cublasH = NULL;
    cudaStream_t stream = NULL;

    const int m = 16;
    const int n = 12;
    const int k = 16;
    const int lda = m;
    const int ldb = k;
    const int ldc = m;

    data_type **matrices = read_bin(m, n, k);

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

//    for (int i = 0; i < m * k; i++) {
//        A.push_back(matrices[0][i]);
//    }
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
//    for (int i = 0; i < n * k; i++) {
//        B.push_back(matrices[1][i]);
//    }


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
        cublasSgemm(cublasH, transa, transb, m, n, k, &alpha, d_A, lda, d_B, ldb, &beta, d_C, ldc));

    /* step 4: copy data to host */
    CUDA_CHECK(cudaMemcpyAsync(C.data(), d_C, sizeof(data_type) * C.size(), cudaMemcpyDeviceToHost,
                               stream));

    CUDA_CHECK(cudaStreamSynchronize(stream));

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