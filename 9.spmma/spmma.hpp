#include<iostream>
#include<fstream>
#include<cuda_runtime_api.h> // cudaMalloc, cudaMemcpy, etc.
#include<cuda_fp16.h>

#include<cstdio>
#include<cstring>       // memset
#include<cstdlib>       // malloc

#include"utils.hpp"

using namespace std;

/*
    src: device IN
    dest: device OUT (need to allocate)
*/
template <typename Dtype>
void __padding_matrix(const Dtype* src, const int row, const int col,
               Dtype *dest, const int row_padding, const int col_padding) {
    cudaMemset(dest, 0, row_padding * col_padding * sizeof(Dtype));
    if (col == col_padding) {
        CUDA_CHECK( cudaMemcpy(dest, src, row * col_padding * sizeof(Dtype), cudaMemcpyDeviceToDevice) )
    } else {
        // spitch指定想要复制的矩阵的本身的宽度 width指定需要复制的宽度 dpitch指定赋值到dest的宽度
        CUDA_CHECK( cudaMemcpy2D(dest, col_padding * sizeof(Dtype), src, col * sizeof(Dtype), col * sizeof(Dtype), row, cudaMemcpyDeviceToDevice) )
    }
}

template <typename Dtype>
__global__ void im2col_gpu_kernel(const int n, const Dtype* data_im, const int data_n, const int channel,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w,
    const int stride_h, const int stride_w,
    const int dilation_h, const int dilation_w,
    const int height_col, const int width_col, Dtype* data_col) {
    CUDA_KERNEL_LOOP(index, n) {
        for (int idn = 0; idn < data_n; idn++) {
            const int h_index = index / width_col;
            const int h_col = h_index % height_col;
            const int w_col = index % width_col;
            const int c_im = h_index / height_col;
            const int c_col = c_im * kernel_h * kernel_w;
            const int h_offset = h_col * stride_h - pad_h;
            const int w_offset = w_col * stride_w - pad_w;
            Dtype* data_col_ptr = data_col;
            data_col_ptr += idn * height_col * width_col + (c_col * height_col * data_n + h_col) * width_col  + w_col;   // 确定输出的pointer的位置
            const Dtype* data_im_ptr = data_im;
            data_im_ptr += idn * channel * height * width + (c_im * height + h_offset) * width + w_offset;   // 确定图像的位置

            for (int i = 0; i < kernel_h; ++i) {
              for (int j = 0; j < kernel_w; ++j) {
                int h_im = h_offset + i * dilation_h;
                int w_im = w_offset + j * dilation_w;
                *data_col_ptr =
                    (h_im >= 0 && w_im >= 0 && h_im < height && w_im < width) ?
                    data_im_ptr[i * dilation_h * width + j * dilation_w] : __int2half_rn(0);
                data_col_ptr += data_n * height_col * width_col;
              }
            }
        }
    }
}

template <typename Dtype>
void im2col_gpu(const Dtype* data_im, const int data_n, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w,
    const int stride_h, const int stride_w,
    const int dilation_h, const int dilation_w, Dtype* data_col) {
    // We are going to launch channels * height_col * width_col kernels, each
    // kernel responsible for copying a single-channel grid.
    int height_col = (height + 2 * pad_h -
      (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
    int width_col = (width + 2 * pad_w -
      (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;
    int num_kernels = channels * height_col * width_col;
    // NOLINT_NEXT_LINE(whitespace/operators)
    im2col_gpu_kernel<Dtype> <<< GET_BLOCKS(num_kernels), CUDA_NUM_THREADS >>>(
        num_kernels, data_im, data_n, channels, height, width, kernel_h, kernel_w, pad_h,
        pad_w, stride_h, stride_w, dilation_h, dilation_w, height_col, width_col, data_col);
    CUDA_POST_KERNEL_CHECK;
}

template <typename Dtype>
__global__ void im2col_rev_kernel(
    const int n, const Dtype *data, int data_n, int kernel_n, int out_h, int out_w, Dtype *out) {
    // row: kernel_n
    // col: n * out_h * out_w
    // n * out_h * out_w个线程
    CUDA_KERNEL_LOOP(index, n) {
        // 每个thread负责一个卷积核对应位置的所有channel
        int line = index % (data_n * out_h * out_w);
        int n_index = index / (out_h * out_w);
        int h_index = (index - n_index * (out_h * out_w)) / out_w;
        int w_index = (index - n_index * (out_h * out_w)) - h_index * out_w;
        for (int i = 0; i < kernel_n; i++) {
            out[n_index * kernel_n * out_h * out_w + i * out_h * out_w + h_index * out_w + w_index] = data[i * data_n * out_h * out_w + line];
        }
    }
}

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
                        printf("%d ", __half2int_rz(tensor[i * c * h * w + j * h * w + k * w + v]));
                    }
                    printf("\n");
                }
            }
        }
        printf("\n");
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

    void copy_from_device(__half *d_data, int size) {
        if (tensor == nullptr) tensor = new __half[size];
        cudaMemcpy(tensor, d_data, size * sizeof(__half), cudaMemcpyDeviceToHost);
    }

};

struct MatrixParam {
    __half *A, *B, *C, *D;
    __half *A_cmpr;
    int *index;
    int m, k, n;

    MatrixParam(__half *A=nullptr, __half *B=nullptr, __half *C=nullptr, __half *D=nullptr, int m=0, int k=0, int n=0):
        A(A), B(B), C(C), D(D), m(m), k(k), n(n) {}

    MatrixParam(int m=0, int k=0, int n=0): A(nullptr), B(nullptr), C(nullptr), D(nullptr), A_cmpr(nullptr), index(nullptr), m(m), k(k), n(n) {}

    void print_matrix(__half *item, int row, int col) {
        for (int i = 0; i < row; i++) {
            for (int j = 0; j < col; j++) {
                printf("%d ", __half2int_rz(item[i * col + j]));
            }
            printf("\n");
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

    int get_m_padding() { return m % 8 ? m + 8 - m % 8 : m; }
    int get_k_padding() { return k % 16 ? k + 16 - k % 16 : k; }
    int get_n_padding() { return n % 8 ? n + 8 - n % 8 : n; }


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

    void matmul_cpu() {
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
        this->D = cpu;
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
                if (abs(static_cast<float>(D[pos]) - static_cast<float>(cpu[pos])) > 0.0001) {
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

    // 对 32x32 的矩阵进行index获取操作

    void transfer_index() {
        
    }
};

struct ConvParam {
    Tensor4d *data, *kernel;
    int padding, stride;

    ConvParam(Tensor4d *data, Tensor4d *kernel, int padding, int stride):
        data(data), kernel(kernel), padding(padding), stride(stride) {}

    int getOut_width() { return (data->w + 2 * padding - kernel->w) / stride + 1; }

    int getOut_height() { return (data->h + 2 * padding - kernel->h) / stride + 1; }

    int getIm2col_size() { return data->n * getOut_height() * getOut_width() * kernel->c * kernel->h * kernel->w; }

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