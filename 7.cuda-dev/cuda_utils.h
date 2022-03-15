#include<cuda_runtime_api.h> // cudaMalloc, cudaMemcpy, etc.
//#include<cusparseLt.h>       // cusparseLt header
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
//const int n_global = 16;
// todo: 大数据测试 多大会崩溃
const int padding_global = 0;
const int stride_global = 1;

const int data_n_global = 4, data_c_global = 3, data_w_global = 16, data_h_global = 16;
const int kernel_n_global = 16, kernel_c_global = 3, kernel_w_global = 3, kernel_h_global = 3;

const int out_w = (data_w_global + 2 * padding_global - kernel_w_global) / stride_global + 1;
const int out_h = (data_h_global + 2 * padding_global - kernel_h_global) / stride_global + 1;

const int m_global = data_n_global * out_w * out_h;
const int k_global = kernel_w_global * kernel_h_global;
const int n_global = kernel_n_global;

//string path = "kernel_" + to_string(kernel_w_global) + "x" + to_string(kernel_w_global) + "/" + to_string(data_w_global) + "x" + to_string(data_w_global) + "/";
string path = "";
string data_path = path + "data.bin";
string kernel_path = path + "kernel.bin";
string a_path = path + "a.bin";
string b_path = path + "b.bin";
string c_path = path + "c.bin";



// ======================================================================= //

void print_matrix(__half *, int, int);

void print_matrix(const int &m, const int &n, const float *A, const int &lda);

void print_tensor(float *item, int n, int c, int h, int w);

void print_tensor(__half *item, int n, int c, int h, int w);

void rand(__half *, int, int);

__half *gemm_cpu(__half *, __half *, int, int, int);

float **read_bin(int data_size, int kernel_size);

float **read_bin(int m, int n, int k);

__half *im2col_data(__half *data, int n, int c, int h, int w, int kernel_h, int kernel_w, int padding, int stride);

// ======================================================================= //

void print_matrix(__half *item, int row, int col) {
    if (row > 100 || col > 100) return;
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

void print_tensor(float *item, int n, int c, int h, int w) {
    if (n * c * w * h > 300) return;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < c; j++) {
            for (int k = 0; k < h; k++) {
                for (int v = 0; v < w; v++) {
                    cout << item[i * c * w * h + j * w * h + k * w + v] << " ";
                }
                cout << endl;
            }
            cout << endl;
        }
        cout << endl;
    }
}

void print_tensor(__half *item, int n, int c, int h, int w) {
    if (n * c * w * h > 300) return;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < c; j++) {
            for (int k = 0; k < h; k++) {
                for (int v = 0; v < w; v++) {
                    cout << item[i * c * w * h + j * w * h + k * w + v] << " ";
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
    memset(ret, 0, sizeof(__half) * m * n);
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

/* 这个data_size和kernel_size 包括了 sizeof(float) */
float **read_bin(int data_size, int kernel_size) {
    float **ret = (float **)malloc(sizeof(float *) * 2);
    float *data = (float *)malloc(data_size);
    float *kernel = (float *)malloc(kernel_size);

    ifstream a_fs(data_path, ios_base::binary);
    a_fs.read((char *)data, data_size);
    ifstream b_fs(kernel_path, ios_base::binary);
    b_fs.read((char *)kernel, kernel_size);

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

__half **read_bin(int data_size, int kernel_size, bool flag) {
    __half **ret = (__half **)malloc(sizeof(__half *) * 2);
    float *data = (float *)malloc(data_size * sizeof(float));
    float *kernel = (float *)malloc(kernel_size * sizeof(float));
    __half *d = new __half[data_size];
    __half *k = new __half[kernel_size];
    ifstream a_fs(data_path, ios_base::binary);
    a_fs.read((char *)data, data_size * sizeof(float));
    ifstream b_fs(kernel_path, ios_base::binary);
    b_fs.read((char *)kernel, kernel_size * sizeof(float));

    for (int i = 0; i < data_size; i++) {
        d[i] = static_cast<__half>(data[i]);
    }

    for (int i = 0; i < kernel_size; i++) {
        k[i] = static_cast<__half>(kernel[i]);
    }
    ret[0] = d;
    ret[1] = k;
    return ret;
}

/* im2col: data为原始数据 会自动padding再进行im2col */
__half *im2col_data(__half *data, int n, int c, int h, int w, int kernel_h, int kernel_w, int padding, int stride) {
    int out_h = (h + 2 * padding - kernel_h) / stride + 1;
    int out_w = (w + 2 * padding - kernel_w) / stride + 1;
    // m = data_n * out_w * out_h, k = kernel_c * kernel_w * kernel_h
    // kernel_c == data_c
    __half *A = (__half *)malloc(sizeof(__half) * n * out_h * out_w * c * kernel_h * kernel_w); 

    // padding
    int data_h_pad = h + padding * 2, data_w_pad = w + padding * 2;
    __half *data_pad = (__half *)malloc(n * c * data_h_pad * data_w_pad * sizeof(__half));
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < c; j++) {
            int index1 = i * c * data_h_pad * data_w_pad + j * data_h_pad * data_w_pad;
            for (int ki = 0; ki < padding; ki++) {
                for (int v = 0; v < data_w_pad; v++) {
                    data_pad[index1 + ki * data_w_pad + v] = static_cast<__half>(static_cast<float>(0));
                }
            }
            for (int ki = padding; ki < padding + h; ki++) {
                for (int v = 0; v < data_w_pad; v++) {
                    if (v < padding || v >= w + padding) data_pad[index1 + ki * data_w_pad + v] = static_cast<__half>(static_cast<float>(0));
                    else data_pad[index1 + ki * data_w_pad + v] = data[i * c * h * w + j * h * w + (ki - padding) * w + v - padding];
                }
            }
            for (int ki = data_h_pad - padding; ki < data_h_pad; ki++) {
                for (int v = 0; v < data_w_pad; v++) {
                    data_pad[index1 + ki * data_w_pad + v] = static_cast<__half>(static_cast<float>(0));
                }
            }
        }
    }
    
    data = data_pad;
    h = data_h_pad;
    w = data_w_pad;
    printf("data_pad: \n-------------\n");
    print_tensor(data, n, c, h, w);

    int cnt = 0;
    for (int ni = 0; ni < n; ni++) {
        int index_n = ni * c * h * w;
        for (int i = 0; i < out_h; i++) {
            for (int j = 0; j < out_w; j++) {
                for (int ci = 0; ci < c; ci++) {
                    int index_c = ci * h * w;
                    int row_num = i * stride, col_num = j * stride;
                    for (int ki = row_num; ki < row_num + kernel_h; ki++) {
                        for (int v = col_num; v < col_num + kernel_w; v++) {
                            if (ki >= h || v >= w) A[cnt++] = static_cast<__half>(static_cast<float>(0));
                            else A[cnt++] = data[index_n + index_c + ki * w + v];
                        }
                    }
                }
            }
        }
    }
    
    printf("im2col: \n");
    print_matrix(A, n * out_h * out_w, c * kernel_h * kernel_w);
    return A;
}

__half *im2col_kernel(__half *kernel, int n, int c, int h, int w) {
    __half *ret = (__half *)malloc(sizeof(__half) * n * c * h * w);
    int k = c * h * w;
    __half *tmp = (__half *)malloc(sizeof(__half) * n * k);
    for (int i = 0; i < n; i++) {
        int cnt = 0;
        for (int j = 0; j < c; j++) {
            for (int ki = 0; ki < h; ki++) {
                for (int v = 0; v < w; v++) {
                    tmp[i * k + cnt++] = kernel[i * c * h * w + j * h * w + ki * w + v];
                }
            }
        }
    }
    int ret_cnt = 0;
    for (int i = 0; i < k; i++) {
        for (int j = 0; j < n; j++) {
            ret[ret_cnt++] = tmp[j * k + i];
        }
    }
    return ret;
}