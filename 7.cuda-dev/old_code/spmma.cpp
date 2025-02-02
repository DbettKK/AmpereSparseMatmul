#include<iostream>
#include<fstream>

#include<cstdint>
#include<cstdio>
#include<cstring>
#include<cstdlib>

#include<cuda_fp16.h>
#include<cuda_runtime_api.h> // cudaMalloc, cudaMemcpy, etc.
#include<cusparseLt.h>       // cusparseLt header

#include"cuda_utils.h"

using namespace std;

int check_gpu();
void cmp_cpu_gpu(__half *gpu, __half *cpu, int m, int n);
Matrix *padding_struct(Matrix *matrix, int flag);   // m->8 n->8 k->16
__half *handle_output(__half *item, int m, int m_pad, int n, int n_pad);    // 将之前padding的进行还原
int calculate(__half *hA, __half *hB, __half *hC, __half *hD, int m, int n, int k); // hc hd都为输出
__half* expose(__half *hA, __half *hB, __half *hC, int m, int n, int k);   // 暴露的接口
__half **convert(float **array, int m, int n, int k);
__half** im2col(__half *data, int data_n, int data_c, int data_h, int data_w, __half *kernel,
            int kernel_n, int kernel_c, int kernel_h, int kernel_w, int stride, int padding);

int main() {
    int m = m_global, k = k_global, n = n_global;
    float **aa = read_bin(m, n, k);
    __half **ret = convert(aa, m, n, k);
    __half *hA = ret[0];
    __half *hB = ret[1];
    __half *hC = ret[2];
    cout << "A:" << endl;
    print_matrix(hA, m, k);
    cout << endl;
    cout << "B:" << endl;
    print_matrix(hB, k, n);
    cout << endl;
    cout << "C:" << endl;
    print_matrix(hC, m, n);
    __half* hD = expose(hA, hB, hC, m, n, k);

}

int main2() {
    int data_n = data_n_global, data_c = data_c_global, data_w = data_w_global, data_h = data_h_global;
    int data_size = data_n * data_c * data_w * data_h;
    int kernel_n = kernel_n_global, kernel_c = kernel_c_global, kernel_w = kernel_w_global, kernel_h = kernel_h_global;
    int kernel_size = kernel_n * kernel_c * kernel_w * kernel_h;
    int padding = padding_global;
    int stride = stride_global;
    int m = m_global, k = k_global, n = n_global;

    __half **ret = read_bin(data_size, kernel_size, true);
    __half **array = im2col(ret[0], data_n, data_c, data_h, data_w, ret[1], kernel_n, kernel_c, kernel_h, kernel_w, padding, stride);

    __half *hA = array[0];
    __half *hB = array[1];
    __half *hC = array[2];
    cout << "A:" << endl;
    print_matrix(hA, m, k);
    cout << endl;
    cout << "B:" << endl;
    print_matrix(hB, k, n);
    cout << endl;
    cout << "C:" << endl;
    print_matrix(hC, m, n);
    cout << endl;
    __half* hD = expose(hA, hB, hC, m, n, k);
    //__half** rev = im2col_rev(hD, data_n, data_c, data_h, data_w, kernel_n, kernel_c, kernel_h, kernel_w, padding, stride);
//    for (int i = 0; i < data_n; i++) {
//        for (int j = 0; j < kernel_n; j++) {
//            printf("================\n");
//            print_matrix(rev[i * kernel_n + j], out_h, out_w);
//        }
//    }
}

int check_gpu() {
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

void cmp_cpu_gpu(__half *gpu, __half *cpu, int m, int n) {
    int total = m * n, cnt = 0;
    printf("total: %d\n", total);

    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            int pos = i * n + j;
            if (gpu[pos] - cpu[pos] > 0.0001) {
                cnt++;
            }
        }
    }
    printf("diff: %d\n", cnt);
}

__half **convert(float **array, int m, int n, int k) {
    __half **ret = (__half **)malloc(sizeof(__half *) * 3);
    __half *ret_a = new __half[m * k];
    __half *ret_b = new __half[k * n];
    __half *ret_c = new __half[m * n];
    for (int i = 0; i < m * k; i++) {
        ret_a[i] = static_cast<__half>(array[0][i]);
    }
    for (int i = 0; i < n * k; i++) {
        ret_b[i] = static_cast<__half>(array[1][i]);
    }
    for (int i = 0; i < m * n; i++) {
        ret_c[i] = static_cast<__half>(array[2][i]);
    }
    ret[0] = ret_a;
    ret[1] = ret_b;
    ret[2] = ret_c;
    return ret;
}

Matrix *padding_struct(Matrix *matrix, int flag) {
    Matrix *out = (Matrix *)malloc(sizeof(Matrix));
    int row = matrix->row, col = matrix->col;
    if (flag == 0) {
        // m * k
        if (row % 8 == 0 && col % 16 == 0) {
            return matrix;
        } else if (row % 8 == 0 && col % 16) {
            int fix = 16 - col % 16;
            __half *tmp = (__half *)malloc(row * (col + fix) * sizeof(__half));
            int cnt = 0;
            for (int i = 0; i < row; i++) {
                for (int j = 0; j < col; j++) {
                    tmp[cnt++] = matrix->item[i * col + j];
                }
                for (int j = 0; j < fix; j++) {
                    tmp[cnt++] = static_cast<__half>(static_cast<float>(0));
                }
            }
            out->item = tmp;
            out->row = row;
            out->col = col + fix;
            return out;
        } else if (row % 8 && col % 16 == 0) {
            int fix = 8 - row % 8;
            __half *tmp = (__half *)malloc((row + fix) * col * sizeof(__half));
            memset(tmp, 0, (row + fix) * col * sizeof(__half));
            memcpy(tmp, matrix->item, row * col * sizeof(__half));
            out->item = tmp;
            out->row = row + fix;
            out->col = col;
            return out;
        } else {
            int fix_row = 8 - row % 8;
            int fix_col = 16 - col % 16;
            __half *tmp = (__half *)malloc((row + fix_row) * (col + fix_col) * sizeof(__half));
            memset(tmp, 0, (row + fix_row) * (col + fix_col) * sizeof(__half));
            int cnt = 0;
            for (int i = 0; i < row; i++) {
                for (int j = 0; j < col; j++) {
                    tmp[cnt++] = matrix->item[i * col + j];
                }
                for (int j = 0; j < fix_col; j++) {
                    tmp[cnt++] = static_cast<__half>(static_cast<float>(0));
                }
            }
            out->item = tmp;
            out->row = row + fix_row;
            out->col = col + fix_col;
            return out;
        }
    } else if (flag == 1) {
        // k * n
        if (row % 16 == 0 && col % 8 == 0) {
            return matrix;
        } else if (row % 16 == 0 && col % 8) {
            int fix = 8 - col % 8;
            __half *tmp = (__half *)malloc(row * (col + fix) * sizeof(__half));
            int cnt = 0;
            for (int i = 0; i < row; i++) {
                for (int j = 0; j < col; j++) {
                    tmp[cnt++] = matrix->item[i * col + j];
                }
                for (int j = 0; j < fix; j++) {
                    tmp[cnt++] = static_cast<__half>(static_cast<float>(0));
                }
            }
            out->item = tmp;
            out->row = row;
            out->col = col + fix;
            return out;
        } else if (row % 16 && col % 8 == 0) {
            int fix = 16 - row % 16;
            __half *tmp = (__half *)malloc((row + fix) * col * sizeof(__half));
            memset(tmp, 0, (row + fix) * col * sizeof(__half));
            memcpy(tmp, matrix->item, row * col * sizeof(__half));
            out->item = tmp;
            out->row = row + fix;
            out->col = col;
            return out;
        } else {
            int fix_row = 16 - row % 16;
            int fix_col = 8 - col % 8;
            __half *tmp = (__half *)malloc((row + fix_row) * (col + fix_col) * sizeof(__half));
            memset(tmp, 0, (row + fix_row) * (col + fix_col) * sizeof(__half));
            int cnt = 0;
            for (int i = 0; i < row; i++) {
                for (int j = 0; j < col; j++) {
                    tmp[cnt++] = matrix->item[i * col + j];
                }
                for (int j = 0; j < fix_col; j++) {
                    tmp[cnt++] = static_cast<__half>(static_cast<float>(0));
                }
            }
            out->item = tmp;
            out->row = row + fix_row;
            out->col = col + fix_col;
            return out;
        }
    } else if (flag == 2) {
        // m * n
        if (row % 8 == 0 && col % 8 == 0) {
            return matrix;
        } else if (row % 8 == 0 && col % 8) {
            int fix = 8 - col % 8;
            __half *tmp = (__half *)malloc(row * (col + fix) * sizeof(__half));
            int cnt = 0;
            for (int i = 0; i < row; i++) {
                for (int j = 0; j < col; j++) {
                    tmp[cnt++] = matrix->item[i * col + j];
                }
                for (int j = 0; j < fix; j++) {
                    tmp[cnt++] = static_cast<__half>(static_cast<float>(0));
                }
            }
            out->item = tmp;
            out->row = row;
            out->col = col + fix;
        } else if (row % 8 && col % 8 == 0) {
            int fix = 8 - row % 8;
            __half *tmp = (__half *)malloc((row + fix) * col * sizeof(__half));
            memset(tmp, 0, (row + fix) * col * sizeof(__half));
            memcpy(tmp, matrix->item, row * col * sizeof(__half));
            out->item = tmp;
            out->row = row + fix;
            out->col = col;
            return out;
        } else {
            int fix_row = 8 - row % 8;
            int fix_col = 8 - col % 8;
            __half *tmp = (__half *)malloc((row + fix_row) * (col + fix_col) * sizeof(__half));
            memset(tmp, 0, (row + fix_row) * (col + fix_col) * sizeof(__half));
            int cnt = 0;
            for (int i = 0; i < row; i++) {
                for (int j = 0; j < col; j++) {
                    tmp[cnt++] = matrix->item[i * col + j];
                }
                for (int j = 0; j < fix_col; j++) {
                    tmp[cnt++] = static_cast<__half>(static_cast<float>(0));
                }
            }
            out->item = tmp;
            out->row = row + fix_row;
            out->col = col + fix_col;
            return out;
        }
    }
}

__half *handle_output(__half *item, int m, int m_pad, int n, int n_pad) {
    if (m_pad == m && n_pad == n) {
        return item;
    }
    __half *ret = (__half *)malloc(m * n * sizeof(__half));
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            ret[i * n + j] = item[i * n_pad + j];
        }
    }
    return ret;
}

int calculate(__half *hA, __half *hB, __half *hC, __half *hD, int m, int n, int k) {
    int A_size = m * k * sizeof(__half);
    int B_size = k * n * sizeof(__half);
    int C_size = m * n * sizeof(__half);

    // Leading dimension 如果行优先则代表列数
    int lda = k, ldb = n, ldc = n;
    auto opA = CUSPARSE_OPERATION_NON_TRANSPOSE;
    auto opB = CUSPARSE_OPERATION_NON_TRANSPOSE;
    float alpha = 1.0f;
    float beta  = 0.0f;

    unsigned alignment = 16;

    auto order = CUSPARSE_ORDER_ROW; // cusparseOrder_t
    auto type  = CUDA_R_16F;
    auto compute_type = CUSPARSE_COMPUTE_16F;

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
    // Prune the A matrix (in-place) and check the correcteness
    // todo: 先check 再prune
    // todo: 测试自己compress的情况
    CHECK_CUSPARSE( cusparseLtSpMMAPrune(&handle, &matmul, dA, dA, CUSPARSELT_PRUNE_SPMMA_TILE, stream) )
    // 这一步可以省略 ↑
    CHECK_CUSPARSE( cusparseLtSpMMAPruneCheck(&handle, &matmul, dA, d_valid, stream) )
    int is_valid;
    CHECK_CUDA( cudaMemcpyAsync(&is_valid, d_valid, sizeof(d_valid), cudaMemcpyDeviceToHost, stream) )
    CHECK_CUDA( cudaStreamSynchronize(stream) )
    if (is_valid != 0) {
        std::printf("!!!! The matrix has been pruned in a wrong way. cusparseLtMatmul will not provide correct results\n");
        return EXIT_FAILURE;
    }
    //--------------------------------------------------------------------------
    // Compress the A matrix
    CHECK_CUSPARSE( cusparseLtSpMMACompressedSize(&handle, &plan, &compressed_size) )
    CHECK_CUDA( cudaMalloc((void**) &dA_compressed, compressed_size) )

    CHECK_CUSPARSE( cusparseLtSpMMACompress(&handle, &plan, dA, dA_compressed, stream) )

    // print to check
    __half *hA_compressed = new __half[compressed_size / sizeof(__half)];
    __half *hA_tmp = new __half[m * k];
    CHECK_CUDA( cudaMemcpy(hA_compressed, dA_compressed, compressed_size, cudaMemcpyDeviceToHost) )
    CHECK_CUDA( cudaMemcpy(hA_tmp, dA, m * k * sizeof(__half), cudaMemcpyDeviceToHost) )
    printf("================================================\n");
    printf("compressed_size: %d\n", compressed_size / sizeof(__half));
    printf("hA: \n");
    print_matrix(hA_tmp, m, k);
    printf("hA_compressed: \n");
    print_matrix(hA_compressed, m, compressed_size / sizeof(__half) / k / 2);
    printf("================================================\n");


    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // Search the best kernel
    void* d_workspace = nullptr;
    int num_streams = 0;
    cudaStream_t* streams = nullptr;

    /*
        The function evaluates all available algorithms for the matrix multiplication and automatically updates the ·plan· by selecting the fastest one.
        The functionality is intended to be used for auto-tuning purposes when the same operation is repeated multiple times over different inputs.
        The function behavior is the same of cusparseLtMatmul().
    */
    //CHECK_CUSPARSE( cusparseLtMatmulSearch(&handle, &plan, &alpha, dA_compressed, dB, &beta, dC,dD, d_workspace, streams, num_streams) )
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
    CHECK_CUDA( cudaMemcpy(hD, dD, C_size, cudaMemcpyDeviceToHost) )
    cout<<" CPU: "<<endl;
    __half *cpu = gemm_cpu(hA, hB, m, n, k);
    print_matrix(cpu, m, n);
    cmp_cpu_gpu(hC, cpu, m, n);
}

__half *expose(__half *hA, __half *hB, __half *hC, int m, int n, int k) {
    check_gpu();

    Matrix *mA = (Matrix *)malloc(sizeof(Matrix));
    Matrix *mB = (Matrix *)malloc(sizeof(Matrix));
    Matrix *mC = (Matrix *)malloc(sizeof(Matrix));
    mA->item = hA;
    mA->row = m;
    mA->col = k;
    mB->item = hB;
    mB->row = k;
    mB->col = n;
    mC->item = hC;
    mC->row = m;
    mC->col = n;
    Matrix *A_out = padding_struct(mA, 0);
    Matrix *B_out = padding_struct(mB, 0);
    Matrix *C_out = padding_struct(mC, 0);

    int m_pad = A_out->row, n_pad = C_out->col, k_pad = A_out->col;

    __half *hD = (__half *)malloc(m_pad * n_pad * sizeof(__half));

    calculate(A_out->item, B_out->item, C_out->item, hD, m_pad, n_pad, k_pad);

    __half *output = handle_output(hD, m, m_pad, n, n_pad);
    cout << "GPU D: " << endl;
    print_matrix(output, m, n);
    return output;
}

__half** im2col(__half *data, int data_n, int data_c, int data_h, int data_w, __half *kernel,
            int kernel_n, int kernel_c, int kernel_h, int kernel_w, int stride, int padding) {
    __half** ret = (__half**)malloc(3 * sizeof(__half*));

    int out_h = (data_h + 2 * padding - kernel_h) / stride + 1;
    int out_w = (data_w + 2 * padding - kernel_w) / stride + 1;

    int m = data_n * out_h * out_w;
    int k = kernel_c * kernel_h * kernel_w;
    int n = kernel_n;

    __half *A = im2col_data(data, data_n, data_c, data_h, data_w, kernel_h, kernel_w, padding, stride);
    __half *B = im2col_kernel(kernel, kernel_n, kernel_c, kernel_h, kernel_w);

    __half *C = (__half*)malloc(m * n * sizeof(__half));
    memset(C, 0, m * n * sizeof(__half));

    ret[0] = A;
    ret[1] = B;
    ret[2] = C;
    return ret;
}


__half** im2col_rev(__half *ans, int data_n, int data_c, int data_h, int data_w,
            int kernel_n, int kernel_c, int kernel_h, int kernel_w, int stride, int padding) {
    int out_h = (data_h + 2 * padding - kernel_h) / stride + 1;
    int out_w = (data_w + 2 * padding - kernel_w) / stride + 1;

    int m = data_n * out_h * out_w;
    int k = kernel_c * kernel_h * kernel_w;
    int n = kernel_n;

    __half **ret = (__half **)malloc(sizeof(__half *) * data_n * kernel_n);
    for (int i = 0; i < data_n * kernel_n; i++) {
        ret[i] = (__half *)malloc(sizeof(__half) * out_h * out_w);
    }
    int cnt_out = 0;
    for (int i = 0; i < data_n; i++) {
        for (int j = 0; j < kernel_n; j++) {
            int cnt_in = 0;
            for (int v = 0; v < out_h * out_w; v++) {
                ret[cnt_out][cnt_in++] = ans[(i * out_h * out_w + v) * kernel_n + j];
            }
            cnt_out++;
        }
    }

    return ret;
}