#include<cuda_runtime_api.h> // cudaMalloc, cudaMemcpy, etc.
#include<cusparseLt.h>       // cusparseLt header
#include<cstdint>
#include<cstdio>
#include<cstring>
#include<cuda_fp16.h>
#include<cstdlib>
#include<fstream>
#include<iostream>

#include<cudnn.h>
#include<cublas_v2.h>


using namespace std;

#define CHECK_CUDA(func)                                                       \
{                                                                              \
    cudaError_t status = (func);                                               \
    if (status != cudaSuccess) {                                               \
        printf("CUDA API failed at line %d with error: %s (%d)\n",             \
               __LINE__, cudaGetErrorString(status), status);                  \
        return EXIT_FAILURE;                                                   \
    }                                                                          \
}

#define CHECK_CUSPARSE(func)                                                   \
{                                                                              \
    cusparseStatus_t status = (func);                                          \
    if (status != CUSPARSE_STATUS_SUCCESS) {                                   \
        printf("CUSPARSE API failed at line %d with error: %s (%d)\n",         \
               __LINE__, cusparseGetErrorString(status), status);              \
        return EXIT_FAILURE;                                                   \
    }                                                                          \
}

#define CHECK_CUDNN(func)                                                      \
{                                                                              \
    cudnnStatus_t status = (func);                                             \
    if (status != CUDNN_STATUS_SUCCESS) {                                      \
        printf("CUDNN failed at line %d with error: %s (%d)\n",                \
               __LINE__, cudnnGetErrorString(status), status);                 \
        return 0;                                                              \
    }                                                                          \
}

#define CHECK_CUBLAS(func)                                                     \
{                                                                              \
    cublasStatus_t status = (func);                                            \
    if (status != CUBLAS_STATUS_SUCCESS) {                                     \
        printf("CUBLAS API failed at line %d with error:  (%d)\n",             \
           __LINE__, status);                                                  \
        return 0;                                                              \
    }                                                                          \
}

struct Matrix {
    __half *item;
    int row, col;
};

constexpr int EXIT_UNSUPPORTED = 2;
//const int m_global = 16;
//const int k_global = 16;
//const int n_global = 12;

const int padding_global = 0;
const int stride_global = 1;

const int data_n_global = 1, data_c_global = 1, data_w_global = 64, data_h_global = 64;
const int kernel_n_global = 64, kernel_c_global = 1, kernel_w_global = 9, kernel_h_global = 9;

const int out_w = (data_w_global + 2 * padding_global - kernel_w_global) / stride_global + 1;
const int out_h = (data_h_global + 2 * padding_global - kernel_h_global) / stride_global + 1;

const int m_global = data_n_global * out_w * out_h;
const int k_global = kernel_w_global * kernel_h_global;
const int n_global = kernel_n_global;

string path = "kernel_" + to_string(kernel_w_global) + "x" + to_string(kernel_w_global) + "/" + to_string(data_w_global) + "x" + to_string(data_w_global) + "/";

string data_path = path + "data.bin";
string kernel_path = path + "kernel.bin";
string a_path = path + "a.bin";
string b_path = path + "b.bin";
string c_path = path + "c.bin";



// ======================================================================= //

void print_matrix(__half *, int, int);

void print_matrix(const int &m, const int &n, const float *A, const int &lda);

void print_tensor(float *item, int n, int c, int w, int h);

void rand(__half *, int, int);

__half *gemm_cpu(__half *, __half *, int, int, int);

float **read_bin(int data_size, int kernel_size);

float **read_bin(int m, int n, int k);

// ======================================================================= //

void print_matrix(__half *item, int row, int col) {
    if (row > 100 && col > 100) return;
    for (int i = 0; i < row; i++) {
        for (int j = 0; j < col; j++) {
            cout << item[i * col + j] << " ";
        }
        cout << endl;
    }
}

void print_matrix(const int &m, const int &n, const float *A, const int &lda) {
    if (lda > 100) return;
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            printf("%0.2f ", A[j * lda + i]);
        }
        printf("\n");
    }
}

void print_tensor(float *item, int n, int c, int w, int h) {
    if (n * c * w * h > 300) return;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < c; j++) {
            for (int k = 0; k < w; k++) {
                for (int v = 0; v < h; v++) {
                    cout << item[i * c * w * h + j * w * h + k * h + v] << " ";
                }
                cout << endl;
            }
            cout << endl;
        }
        cout << endl;
    }
}

void rand(__half *item, int m, int n) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            item[i * n + j] = static_cast<__half>(static_cast<float>(rand() % 8));
        }
    }
}

__half *gemm_cpu(__half *A, __half *B, int m, int n, int k) {
    __half *ret = (__half *)malloc(sizeof(__half) * m * n);
    memset(ret, 0, sizeof(m * n));
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            float sum  = 0.0f;
            for (int v = 0; v < k; v++) {
                int posA =  i * k + v; // A[i][v]
                int posB =  v * n + j; // B[v][j]
                sum += static_cast<float>(A[posA]) * static_cast<float>(B[posB]);
            }
            int posRet = i * n + j;
            ret[posRet] = sum;  // [i][j]
        }
    }
    return ret;
}

float **read_bin(int data_size, int kernel_size) {
    float **ret = (float **)malloc(sizeof(float *) * 2);
    float *data = (float *)malloc(data_size);
    float *kernel = (float *)malloc(kernel_size);

    ifstream a_fs(data_path, ios_base::binary);
    a_fs.read((char *)data, data_size);
    ifstream b_fs(kernel_path, ios_base::binary);
    b_fs.read((char *)kernel, kernel_size);

    for (int i = 0; i < data_size / sizeof(float); i++) {
        cout << data[i] << " ";
    }
    cout << endl;

    ret[0] = data;
    ret[1] = kernel;
    return ret;
}

float **read_bin(int m, int n, int k) {
    float **ret = (float **)malloc(sizeof(float *) * 3);
    float *mat_a_host = new float[m * k];
    float *mat_b_host = new float[k * n];
    float *mat_c_host = new float[m * n];
    ifstream a_fs(a_path, ios_base::binary);
    a_fs.read((char *)mat_a_host, m * k * sizeof(float));
    ifstream b_fs(b_path, ios_base::binary);
    b_fs.read((char *)mat_b_host, k * n * sizeof(float));
    ifstream c_fs(c_path, ios_base::binary);
    c_fs.read((char *)mat_c_host, m * n * sizeof(float));

    ret[0] = mat_a_host;
    ret[1] = mat_b_host;
    ret[2] = mat_c_host;
    return ret;
}