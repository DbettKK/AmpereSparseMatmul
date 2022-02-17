#include<cstdint>
#include<cstdio>
#include<cstring>
#include<cuda_fp16.h>
#include<fstream>
#include<iostream>

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
    hA = handle_input(hA, m, n);
    hB = handle_input(hB, n, k);
    hC = handle_input(hC, m, k);

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

__half *handle_input(__half *item, int m, int n) {
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
        return ret;
    }
    if (n % 8 == 0) {
        int fix = 8 - m % 8;
        __half *ret = (__half *)malloc((m + fix) * n * sizeof(__half));
        memset(ret, 0, (m + fix) * n * sizeof(__half));
        memcpy(ret, item, m * n * sizeof(__half));
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
    return ret;
}

void calculate() {

}

void print() {
}

int main() {
    init();
    input();
    handle_input();
    calculate();
    print();
}