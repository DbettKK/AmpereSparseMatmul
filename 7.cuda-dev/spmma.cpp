#include<cuda_runtime_api.h> // cudaMalloc, cudaMemcpy, etc.
#include<cusparseLt.h>       // cusparseLt header
#include<cstdint>
#include<cstdio>
#include<cstring>
#include<cuda_fp16.h>
#include<cstdlib>
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

#define CHECK_CUSPARSE(func)                                                   \
{                                                                              \
    cusparseStatus_t status = (func);                                          \
    if (status != CUSPARSE_STATUS_SUCCESS) {                                   \
        printf("CUSPARSE API failed at line %d with error: %s (%d)\n",         \
               __LINE__, cusparseGetErrorString(status), status);              \
        return EXIT_FAILURE;                                                   \
    }                                                                          \
}

struct Matrix {
    __half *item;
    int row, col;
};

constexpr int EXIT_UNSUPPORTED = 2;

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

void print_matrix(__half *item, int row, int col) {
    for (int i = 0; i < row; i++) {
        for (int j = 0; j < col; j++) {
            cout << item[i * col + j] << " ";
        }
        cout << endl;
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

void cmp_cpu_gpu(__half *gpu, __half *cpu, int m, int n) {
    int total = m * n, cnt = 0;
    printf("total: %d\n", total);

    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            int pos = i * n + j;
            if (gpu[pos] - cpu[pos] > 0.01) {
                cnt++;
            }
        }
    }
    printf("diff: %d\n", cnt);
}

// m->8 n->8 k->16
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

// 将之前padding的进行还原
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

// 在整体较大时 需要进行tile
void tile(__half *item, int row, int col) {

}

__half **read_bin(int m, int n, int k) {
    __half **ret = (__half **)malloc(sizeof(__half *) * 3);
    __half *mat_a_host = new __half[m * k];
    __half *mat_b_host = new __half[k * n];
    __half *mat_c_host = new __half[m * n];
    ifstream a_fs("a.bin", ios_base::binary);
    a_fs.read((char *)mat_a_host, m * k * sizeof(__half));
    ifstream b_fs("b.bin", ios_base::binary);
    b_fs.read((char *)mat_b_host, k * n * sizeof(__half));
    ifstream c_fs("c.bin", ios_base::binary);
    c_fs.read((char *)mat_c_host, m * n * sizeof(__half));
    ret[0] = mat_a_host;
    ret[1] = mat_b_host;
    ret[2] = mat_c_host;
    return ret;
}

// hc hd都为输出
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
    //CHECK_CUSPARSE( cusparseLtSpMMAPrune(&handle, &matmul, dA, dA, CUSPARSELT_PRUNE_SPMMA_TILE, stream) )
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
    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // Search the best kernel
    void* d_workspace = nullptr;
    int num_streams = 0;
    cudaStream_t* streams = nullptr;
    CHECK_CUSPARSE( cusparseLtMatmulSearch(&handle, &plan, &alpha, dA_compressed, dB, &beta, dC,dD, d_workspace, streams, num_streams) )
    int alg_id;
    CHECK_CUSPARSE( cusparseLtMatmulAlgGetAttribute(&handle, &alg_sel, CUSPARSELT_MATMUL_ALG_CONFIG_ID, &alg_id, sizeof(alg_id)) )
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
    CHECK_CUDA( cudaMemcpy(hD, dD, C_size, cudaMemcpyDeviceToHost) )
    cout << "A_compress: " << endl;
    //print_matrix(hA, m, n);
    cout<<" CPU: "<<endl;
    __half *cpu = gemm_cpu(hA, hB, m, n, k);
    //print_matrix(cpu, m, n);
    cmp_cpu_gpu(hC, cpu, m, n);
}

void expose(__half *hA, __half *hB, __half *hC, int m, int n, int k) {
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
    //print_matrix(output, m, n);
}

void rand(__half *item, int m, int n) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            item[i * n + j] = static_cast<__half>(static_cast<float>(rand() % 8));
        }
    }
}

int main() {
    int m = 512, k = 512, n = 512;
    __half **array = read_bin(m, n, k);
//    __half *hA = (__half *)malloc(m * k * sizeof(__half));
//    __half *hB = (__half *)malloc(k * n * sizeof(__half));
//    __half *hC = (__half *)malloc(m * n * sizeof(__half));
//    rand(hA, m, k);
//    rand(hB, k, n);
//    memset(hC, 0, m * n * sizeof(__half));
    //rand(hC, m, n);
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
    expose(hA, hB, hC, m, n, k);
}
