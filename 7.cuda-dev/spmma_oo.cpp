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

using namespace std;

using spmmaStatus_t = int;

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

const spmmaStatus_t SUCCESS = 0;
const spmmaStatus_t DO_NOTHING = 1;
const spmmaStatus_t ERROR = 2;
const spmmaStatus_t UNSUPPORTED = 3;

struct Tensor4d {
    __half *tensor;
    int n, c, h, w;
    Tensor4d(__half *tensor, int n, int c, int h, int w): tensor(tensor), n(n), c(c), h(h), w(w) {}
    Tensor4d(int n, int c, int h, int w): tensor(nullptr), n(n), c(c), h(h), w(w) {}

    int get_size() {
        return n * c * h * w;
    }

    void print_tensor() {
        for (int i = 0; i < n; i++) {
            printf("n%d:\n", i);
            for (int j = 0; j < c; j++) {
                printf("c%d:\n", j);
                for (int k = 0; k < h; k++) {
                    for (int v = 0; v < w; v++) {
                        cout << tensor[i * c * h * w + j * h * w + k * w + v] << " ";
                    }
                    printf("\n");
                }
            }
        }
        cout << endl;
    }

    void read_bin(string path) {
        float *bin_file = new float[get_size()];
        ifstream a_fs(path, ios_base::binary);
        a_fs.read((char *)bin_file, get_size() * sizeof(float));
        if (tensor == nullptr) tensor = new __half[get_size()];
        for (int i = 0; i < get_size(); i++) {
            tensor[i] = static_cast<__half>(bin_file[i]);
        }
    }

    void generate_rand(int bound) {
        if (tensor == nullptr) tensor = new __half[get_size()];
        for (int i = 0; i < get_size(); i++) {
            tensor[i] = static_cast<__half>(static_cast<float>(rand() % bound));
        }
    }
};

struct MatrixParam {
    __half *A, *B, *C, *D;
    int m, k, n;
    MatrixParam(__half *A=nullptr, __half *B=nullptr, __half *C=nullptr, __half *D=nullptr, int m=0, int k=0, int n=0):
        A(A), B(B), C(C), D(D), m(m), k(k), n(n) {}

    MatrixParam(int m=0, int k=0, int n=0): A(nullptr), B(nullptr), C(nullptr), D(nullptr), m(m), k(k), n(n) {}

    void print_matrix(__half *item, int row, int col) {
        for (int i = 0; i < row; i++) {
            for (int j = 0; j < col; j++) {
                cout << item[i * col + j] << " ";
            }
            cout << endl;
        }
    }

    void print_all() {
        printf("m: %d, k: %d, n: %d\n", this->m, this->k, this->n);
        printf("A:\n");
        print_matrix(this->A, this->m, this->k);
        printf("B:\n");
        print_matrix(this->B, this->k, this->n);
        printf("C:\n");
        print_matrix(this->C, this->m, this->n);
        printf("D:\n");
        print_matrix(this->D, this->m, this->n);
    }

    void copy(MatrixParam *param) {
        this->A = param->A;
        this->B = param->B;
        this->C = param->C;
        this->D = param->D;
        this->m = param->m;
        this->k = param->k;
        this->n = param->n;
    }

    MatrixParam *fix_matrix() {
        // get the number row/col need to pad
        int fix_m = m % 8 ? 8 - m % 8 : 0;
        int fix_k = k % 16 ? 16 - k % 16 : 0;
        int fix_n = n % 8 ? 8 - n % 8 : 0;
        __half *A_new = new __half[(m + fix_m) * (k + fix_k)];
        __half *B_new = new __half[(k + fix_k) * (n + fix_n)];
        __half *C_new = new __half[(m + fix_m) * (n + fix_n)];
        __half *D_new = new __half[(m + fix_m) * (n + fix_n)];
        memset(A_new, 0, (m + fix_m) * (k + fix_k) * sizeof(__half));
        memset(B_new, 0, (k + fix_k) * (n + fix_n) * sizeof(__half));
        memset(C_new, 0, (m + fix_m) * (n + fix_n) * sizeof(__half));
        memset(D_new, 0, (m + fix_m) * (n + fix_n) * sizeof(__half));
        // padding
        int cnt_A = 0, cnt_B = 0, cnt_C = 0;
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < k + fix_k; j++) {
                if (j < k) A_new[cnt_A++] = A[i * k + j];
                else A_new[cnt_A++] = static_cast<__half>(static_cast<float>(0));
            }
            for (int j = 0; j < n + fix_n; j++) {
                if (j < n) C_new[cnt_C++] = C[i * n + j];
                else C_new[cnt_C++] = static_cast<__half>(static_cast<float>(0));
            }
        }
        for (int i = 0; i < k; i++) {
            for (int j = 0; j < n + fix_n; j++) {
                if (j < n) B_new[cnt_B++] = B[i * n + j];
                else B_new[cnt_B++] = static_cast<__half>(static_cast<float>(0));
            }
        }
        return new MatrixParam(A_new, B_new, C_new, D_new, m + fix_m, k + fix_k, n + fix_n);
    }

    MatrixParam *refix_matrix(int m_old, int n_old) {
        if (m_old == m && n_old == n) {
            return this;
        }
        __half *ret = new __half[m_old * n_old];
        for (int i = 0; i < m_old; i++) {
            for (int j = 0; j < n_old; j++) {
                ret[i * n_old + j] = D[i * n_old + j];
            }
        }
        return new MatrixParam(A, B, C, ret, m, k, n);
    }

    Tensor4d *im2col_rev(int data_n, int kernel_n, int out_h, int out_w) {
        __half *ret = new __half[data_n * kernel_n * out_h * out_w];
        int cnt = 0;
        for (int i = 0; i < data_n; i++) {
            for (int j = 0; j < kernel_n; j++) {
                int cnt_in = 0;
                for (int v = 0; v < out_h * out_w; v++) {
                    ret[cnt++] = D[(i * out_h * out_w + v) * kernel_n + j];
                }
            }
        }

        return new Tensor4d(ret, data_n, kernel_n, out_h, out_w);
    }

    bool check_correct() {
        // cpu
        __half *cpu = new __half[m * n];
        memset(cpu, 0, m * n * sizeof(__half));
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                float sum  = 0.0f;
                for (int v = 0; v < k; v++) {
                    int posA =  i * k + v; // A[i][v]
                    int posB =  v * n + j; // B[v][j]
                    sum += static_cast<float>(A[posA]) * static_cast<float>(B[posB]);
                }
                int posRet = i * n + j;
                cpu[posRet] = sum;  // [i][j]
            }
        }
        printf("cpu:\n");
        print_matrix(cpu, m, n);
        // diff
        printf("check cpu with gpu:\n");
        int total = m * n, cnt = 0;
        printf("total: %d\n", total);
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                int pos = i * n + j;
                if (abs(D[pos] - cpu[pos]) > 0.0001) {
                    cnt++;
                }
            }
        }
        printf("diff: %d\n", cnt);
        printf("gpu:\n");
        print_matrix(D, m, n);
        return cnt == 0;
    }

    void read_bin(string matA_path, string matB_path, string matC_path) {
        float *matA = new float[m * k];
        float *matB = new float[k * n];
        float *matC = new float[m * n];

        ifstream a_fs(matA_path, ios_base::binary);
        a_fs.read((char *)matA, m * k * sizeof(float));
        ifstream b_fs(matB_path, ios_base::binary);
        b_fs.read((char *)matB, k * n * sizeof(float));
        ifstream c_fs(matC_path, ios_base::binary);
        c_fs.read((char *)matC, m * n * sizeof(float));

        if (A == nullptr)  A = new __half[m * k];
        if (B == nullptr)  B = new __half[k * n];
        if (C == nullptr)  C = new __half[m * n];

        for (int i = 0; i < m * k; i++) {
            A[i] = static_cast<__half>(matA[i]);
        }
        for (int i = 0; i < k * n; i++) {
            B[i] = static_cast<__half>(matB[i]);
        }
        for (int i = 0; i < m * n; i++) {
            C[i] = static_cast<__half>(matC[i]);
        }
    }

    void generate_rand(int bound) {
        if (A == nullptr)  A = new __half[m * k];
        if (B == nullptr)  B = new __half[k * n];
        if (C == nullptr)  C = new __half[m * n];

        for (int i = 0; i < m * k; i++) {
            A[i] = static_cast<__half>(static_cast<float>(rand() % bound));
        }
        for (int i = 0; i < k * n; i++) {
            B[i] = static_cast<__half>(static_cast<float>(rand() % bound));
        }
        for (int i = 0; i < m * n; i++) {
            C[i] = static_cast<__half>(static_cast<float>(0));
        }
    }

    __half* generate_sparse_cmpr(int bound) {
        __half *ret = new __half[k * n / 2];
        if (A == nullptr)  A = new __half[m * k];
        if (B == nullptr)  B = new __half[k * n];
        if (C == nullptr)  C = new __half[m * n];

        for (int i = 0; i < m * k; i++) {
            A[i] = static_cast<__half>(static_cast<float>(rand() % bound));
        }
        // 四个为一组 每一组如果前两个为0 则会忽略
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < k; j+=2) {
                int zero_index = rand() % 2;
                B[(j + zero_index) * n + i] = static_cast<__half>(static_cast<float>(rand() % bound + 1));
                B[(j + 1 - zero_index) * n + i] = B[(j + 1 - zero_index) * n + i] = static_cast<__half>(static_cast<float>(0));
                ret[j / 2 * n + i] = B[(j + zero_index) * n + i];
            }
        }
        for (int i = 0; i < m * n; i++) {
            C[i] = static_cast<__half>(static_cast<float>(0));
        }
        return ret;
    }

    __half* generate_sparse_cmpr_A(int bound) {
        int ret_cnt = 0;
        __half *ret = new __half[m * k / 2];
        if (A == nullptr)  A = new __half[m * k];
        if (B == nullptr)  B = new __half[k * n];
        if (C == nullptr)  C = new __half[m * n];

        // 四个为一组 每一组如果前两个为0 则会忽略
        for (int i = 0; i < m * k; i+=2) {
            int zero_index = rand() % 2;
            A[i + zero_index] = static_cast<__half>(static_cast<float>(rand() % bound + 1));
            A[i + 1 - zero_index] = static_cast<__half>(static_cast<float>(0));
            ret[ret_cnt++] = A[i + zero_index];
        }
        for (int i = 0; i < k * n; i++) {
            B[i] = static_cast<__half>(static_cast<float>(rand() % bound));
        }
        for (int i = 0; i < m * n; i++) {
            C[i] = static_cast<__half>(static_cast<float>(0));
        }
        return ret;
    }
};

struct ConvParam {
    Tensor4d *data, *kernel;
    int padding, stride;

    ConvParam(Tensor4d *data, Tensor4d *kernel, int padding, int stride):
        data(data), kernel(kernel), padding(padding), stride(stride) {}

    int getOut_width() { return (data->w + 2 * padding - kernel->w) / stride + 1; }

    int getOut_height() { return (data->h + 2 * padding - kernel->h) / stride + 1; }

    Tensor4d *pad_data() {
        int data_h_pad = data->h + padding * 2, data_w_pad = data->w + padding * 2;
        __half *data_pad = new __half[data->n * data->c * data_h_pad * data_w_pad];
        for (int i = 0; i < data->n; i++) {
            for (int j = 0; j < data->c; j++) {
                int index1 = i * data->c * data_h_pad * data_w_pad + j * data_h_pad * data_w_pad;
                for (int ki = 0; ki < padding; ki++) {
                    for (int v = 0; v < data_w_pad; v++) {
                        data_pad[index1 + ki * data_w_pad + v] = static_cast<__half>(static_cast<float>(0));
                    }
                }
                for (int ki = padding; ki < padding + data->h; ki++) {
                    for (int v = 0; v < data_w_pad; v++) {
                        if (v < padding || v >= data->w + padding) data_pad[index1 + ki * data_w_pad + v] = static_cast<__half>(static_cast<float>(0));
                        else data_pad[index1 + ki * data_w_pad + v] = data->tensor[i * data->c * data->h * data->w + j * data->h * data->w + (ki - padding) * data->w + v - padding];
                    }
                }
                for (int ki = data_h_pad - padding; ki < data_h_pad; ki++) {
                    for (int v = 0; v < data_w_pad; v++) {
                        data_pad[index1 + ki * data_w_pad + v] = static_cast<__half>(static_cast<float>(0));
                    }
                }
            }
        }
        return new Tensor4d(data_pad, data->n, data->c, data_h_pad, data_w_pad);
    }

    __half *im2col_data() {
        // m = data_n * out_w * out_h, k = kernel_c * kernel_w * kernel_h
        // kernel_c == data_c
        int out_h = getOut_height(), out_w = getOut_width();
        __half *ret = (__half *)malloc(sizeof(__half) * data->n * out_h * out_w * kernel->c * kernel->h * kernel->w);

        // padding
        Tensor4d *data_pad = pad_data();

        // im2col4d
        int cnt = 0;
        for (int ni = 0; ni < data_pad->n; ni++) {
            int index_n = ni * data_pad->c * data_pad->h * data_pad->w;
            for (int i = 0; i < out_h; i++) {
                for (int j = 0; j < out_w; j++) {
                    for (int ci = 0; ci < data_pad->c; ci++) {
                        int index_c = ci * data_pad->h * data_pad->w;
                        int row_num = i * stride, col_num = j * stride;
                        for (int ki = row_num; ki < row_num + kernel->h; ki++) {
                            for (int v = col_num; v < col_num + kernel->w; v++) {
                                if (ki >= data_pad->h || v >= data_pad->w)
                                    ret[cnt++] = static_cast<__half>(static_cast<float>(0));
                                else
                                    ret[cnt++] = data_pad->tensor[index_n + index_c + ki * data_pad->w + v];
                            }
                        }
                    }
                }
            }
        }
        return ret;
    }

    __half *im2col_kernel() {
        __half *ret = new __half[kernel->n * kernel->c * kernel->h * kernel->w];
        int k = kernel->c * kernel->h * kernel->w;
        __half *tmp = new __half [k * kernel->n];
        for (int i = 0; i < kernel->n; i++) {
            int cnt = 0;
            for (int j = 0; j < kernel->c; j++) {
                for (int ki = 0; ki < kernel->h; ki++) {
                    for (int v = 0; v < kernel->w; v++) {
                        tmp[i * k + cnt++] = kernel->tensor[i * kernel->c * kernel->h * kernel->w + j * kernel->h * kernel->w + ki * kernel->w + v];
                    }
                }
            }
        }
        int ret_cnt = 0;
        for (int i = 0; i < k; i++) {
            for (int j = 0; j < kernel->n; j++) {
                ret[ret_cnt++] = tmp[j * k + i];
            }
        }
        return ret;
    }

    MatrixParam *im2col() {
        int out_h = getOut_height(), out_w = getOut_width();

        int m = data->n * out_h * out_w;
        int k = kernel->c * kernel->h * kernel->w;
        int n = kernel->n;

        __half *A = im2col_data();
        __half *B = im2col_kernel();

        __half *C = new __half[m * n];
        __half *D = new __half[m * n];
        memset(C, 0, m * n * sizeof(__half));
        memset(D, 0, m * n * sizeof(__half));

        return new MatrixParam(A, B, C, D, m, k, n);
    }

    Tensor4d *im2col_rev(MatrixParam *param) {
        int out_h = getOut_height(), out_w = getOut_width();
        __half *ans = param->D;
        __half *ret = new __half[data->n * kernel->n * out_h * out_w];

        int cnt = 0;
        for (int i = 0; i < data->n; i++) {
            for (int j = 0; j < kernel->n; j++) {
                int cnt_in = 0;
                for (int v = 0; v < out_h * out_w; v++) {
                    ret[cnt++] = ans[(i * out_h * out_w + v) * kernel->n + j];
                }
            }
        }

        return new Tensor4d(ret, data->n, kernel->n, out_h, out_w);
    }
};

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

spmmaStatus_t __mma_matmul(MatrixParam *param, __half *matB_cmpr) {
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

//    CHECK_CUSPARSE( cusparseLtSpMMAPruneCheck(&handle, &matmul, dB, d_valid, stream) )
//    int is_valid;
//    CHECK_CUDA( cudaMemcpyAsync(&is_valid, d_valid, sizeof(d_valid), cudaMemcpyDeviceToHost, stream) )
//    CHECK_CUDA( cudaStreamSynchronize(stream) )
//    if (is_valid != 0) {
//        std::printf("!!!! The matrix need to be pruned.\n");
//        CHECK_CUSPARSE( cusparseLtSpMMAPrune(&handle, &matmul, dB, dB, CUSPARSELT_PRUNE_SPMMA_TILE, stream) )
//    }
//    // 需要把prune后的b拿出来
//    __half *newB = new __half[k * n];
//    CHECK_CUDA( cudaMemcpy(newB, dB, B_size, cudaMemcpyDeviceToHost) )
//    param->B = newB;
//    // Compress the A matrix
    CHECK_CUSPARSE( cusparseLtSpMMACompressedSize(&handle, &plan, &compressed_size) )
    CHECK_CUDA( cudaMalloc((void**) &dB_compressed, compressed_size) )
    CHECK_CUSPARSE( cusparseLtSpMMACompress(&handle, &plan, dB, dB_compressed, stream) )

//    __half *hB_compressed = new __half[compressed_size / sizeof(__half)];
//    __half *hB_tmp = new __half[k * n];
//    CHECK_CUDA( cudaMemcpy(hB_compressed, dB_compressed, compressed_size, cudaMemcpyDeviceToHost) )
//    CHECK_CUDA( cudaMemcpy(hB_tmp, dB, k * n * sizeof(__half), cudaMemcpyDeviceToHost) )
//    printf("================================================\n");
//    printf("compressed_size: %d\n", compressed_size / sizeof(__half));
//    printf("%d\n", sizeof(__half));
//    printf("hB: \n");
//    param->print_matrix(hB_tmp, k, n);
//    printf("hB_compressed: \n");
//    param->print_matrix(hB_compressed, k, n);
//    printf("================================================\n");

//        cout << "B: " << endl;
//        param->print_matrix(param->B, k, n);
//        cout << "B_cmpr: " << endl;
//        param->print_matrix(matB_cmpr, k / 2, n);

//        CHECK_CUSPARSE( cusparseLtSpMMAPruneCheck(&handle, &matmul, dB, d_valid, stream) )
//        int is_valid;
//        CHECK_CUDA( cudaMemcpyAsync(&is_valid, d_valid, sizeof(d_valid), cudaMemcpyDeviceToHost, stream) )
//        CHECK_CUDA( cudaStreamSynchronize(stream) )
//        if (is_valid != 0) {
//            std::printf("!!!! The matrix need to be pruned.\n");
//            //CHECK_CUSPARSE( cusparseLtSpMMAPrune(&handle, &matmul, dB, dB, CUSPARSELT_PRUNE_SPMMA_TILE, stream) )
//        }
//        CHECK_CUSPARSE( cusparseLtSpMMACompressedSize(&handle, &plan, &compressed_size) )
//        CHECK_CUDA( cudaMalloc((void**) &dB_compressed, compressed_size) )
//        cout << compressed_size / sizeof(void) << endl;
//        CHECK_CUSPARSE( cusparseLtSpMMACompress(&handle, &plan, dB, dB_compressed, stream) )
//        __half *hB_compressed = new __half[compressed_size / sizeof(__half)];
//        CHECK_CUDA( cudaMemcpy(hB_compressed, dB_compressed, compressed_size, cudaMemcpyDeviceToHost) )
//        cout << "GPU_cmpr: " << endl;
//        for (int i = 0; i < compressed_size / sizeof(__half); i++) {
//            cout << hB_compressed[i] << " ";
//        }
//        cout << endl;
//
//        cout << "cs: " << compressed_size << endl;
//        cout << "me,cs: " << k * n / 2 * sizeof(__half) << endl;
//        compressed_size = k * n * sizeof(__half);
//        CHECK_CUDA( cudaMalloc((void**) &dB_compressed, compressed_size) )
//        CHECK_CUDA( cudaMemcpyAsync(dB_compressed, matB_cmpr, compressed_size, cudaMemcpyHostToDevice, stream) )

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

    // todo: 测试 matA情况下的cmprsize问题以及matA16816的速度
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
MatrixParam* spmma_matmul(MatrixParam *param, __half *matB_cmpr) {
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

    out->print_all();

    // 2.calculate
    __mma_matmul(out, matB_cmpr);

    // 3. compare with cpu
    out->check_correct();

    return out;
}

Tensor4d *spmma_conv(ConvParam *param) {
    MatrixParam *matrix = param->im2col();  // 最初版本的matrix
    MatrixParam *ans = spmma_matmul(matrix, nullptr);   // 这是fix后并且计算了D的matrix
    MatrixParam *refix = ans->refix_matrix(matrix->m, matrix->n);    // 是把D重新恢复的matrix 其他都不变
    Tensor4d *ret = param->im2col_rev(refix);
    return ret;
}

void test_gemm(int m, int k, int n) {
    MatrixParam *param = new MatrixParam(m, k, n);
    __half *cmpr = param->generate_sparse_cmpr(5);
    MatrixParam *ans = spmma_matmul(param, cmpr);
    //ans->check_correct();
    // compress b的时候 是反过来的
    //ans->check_correct();
}

void test_conv() {
    Tensor4d *data = new Tensor4d(1, 1, 6, 6);
    Tensor4d *kernel = new Tensor4d(2, 3, 3, 3);

    data->generate_rand(5);
    kernel->generate_rand(3);
    data->print_tensor();
    kernel->print_tensor();
    Tensor4d *ans = spmma_conv(new ConvParam(data, kernel, 0, 1));
    ans->print_tensor();
}

int main() {
    test_conv();
}



// todo: tile还未考虑
// todo: cpu时间的考虑