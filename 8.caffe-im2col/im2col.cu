#include<algorithm>
#include<iostream>
#include<cuda_fp16.h>
#include "im2col.hpp"
using namespace std;

// CUDA: grid stride looping
#define CUDA_KERNEL_LOOP(i, n) \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n);  i += blockDim.x * gridDim.x)


#define CUDA_CHECK(func)                                                       \
{                                                                              \
    cudaError_t status = (func);                                               \
    if (status != cudaSuccess) {                                               \
        printf("CUDA API failed at line %d with error: %s (%d)\n",             \
               __LINE__, cudaGetErrorString(status), status);                  \
    }                                                                          \
}

//#define CUDA_CHECK(condition) \
//  /* Code block avoids redefinition of cudaError_t error */ \
//  do { \
//    cudaError_t error = condition; \
//    CHECK_EQ(error, cudaSuccess) << " " << cudaGetErrorString(error); \
//  } while (0)

#define CUDA_POST_KERNEL_CHECK CUDA_CHECK(cudaPeekAtLastError())

// CUDA: use 512 threads per block
const int CAFFE_CUDA_NUM_THREADS = 512;

// CUDA: number of blocks for threads.
inline int CAFFE_GET_BLOCKS(const int N) {
  return (N + CAFFE_CUDA_NUM_THREADS - 1) / CAFFE_CUDA_NUM_THREADS;
}


template <typename Dtype>
__global__ void im2col_gpu_kernel(const int n, const Dtype* data_im, const int data_n, const int channel,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w,
    const int stride_h, const int stride_w,
    const int dilation_h, const int dilation_w,
    const int height_col, const int width_col,
    Dtype* data_col) {
    CUDA_KERNEL_LOOP(index, n) {
    // 很多个n
    for (int idn = 0; idn < data_n; idn++){
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
                data_im_ptr[i * dilation_h * width + j * dilation_w] : 0;
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
    const int dilation_h, const int dilation_w,
    Dtype* data_col) {
  // We are going to launch channels * height_col * width_col kernels, each
  // kernel responsible for copying a single-channel grid.
  int height_col = (height + 2 * pad_h -
      (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
  int width_col = (width + 2 * pad_w -
      (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;
  int num_kernels = channels * height_col * width_col;
  // NOLINT_NEXT_LINE(whitespace/operators)
  im2col_gpu_kernel<Dtype> <<< CAFFE_GET_BLOCKS(num_kernels), CAFFE_CUDA_NUM_THREADS >>>(
      num_kernels, data_im, data_n, channels, height, width, kernel_h, kernel_w, pad_h,
      pad_w, stride_h, stride_w, dilation_h, dilation_w, height_col,
      width_col, data_col);
  CUDA_POST_KERNEL_CHECK;
}

// Explicit instantiation
template void im2col_gpu<float>(const float* data_im, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w, const int stride_h, const int stride_w,
    const int dilation_h, const int dilation_w, float* data_col);
template void im2col_gpu<double>(const double* data_im, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w, const int stride_h, const int stride_w,
    const int dilation_h, const int dilation_w, double* data_col);
template void im2col_gpu<__half>(const __half* data_im, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w, const int stride_h, const int stride_w,
    const int dilation_h, const int dilation_w, __half* data_col);

template <typename Dtype>
__global__ void col2im_gpu_kernel(const int n, const Dtype* data_col,
    const int height, const int width, const int channels,
    const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w,
    const int stride_h, const int stride_w,
    const int dilation_h, const int dilation_w,
    const int height_col, const int width_col,
    Dtype* data_im) {
  CUDA_KERNEL_LOOP(index, n) {
    Dtype val = 0;
    const int w_im = index % width + pad_w;
    const int h_im = (index / width) % height + pad_h;
    const int c_im = index / (width * height);
    int kernel_extent_w = (kernel_w - 1) * dilation_w + 1;
    int kernel_extent_h = (kernel_h - 1) * dilation_h + 1;
    // compute the start and end of the output
    const int w_col_start =
        (w_im < kernel_extent_w) ? 0 : (w_im - kernel_extent_w) / stride_w + 1;
    const int w_col_end = min(w_im / stride_w + 1, width_col);
    const int h_col_start =
        (h_im < kernel_extent_h) ? 0 : (h_im - kernel_extent_h) / stride_h + 1;
    const int h_col_end = min(h_im / stride_h + 1, height_col);
    // TODO: use LCM of stride and dilation to avoid unnecessary loops
    for (int h_col = h_col_start; h_col < h_col_end; h_col += 1) {
      for (int w_col = w_col_start; w_col < w_col_end; w_col += 1) {
        int h_k = (h_im - h_col * stride_h);
        int w_k = (w_im - w_col * stride_w);
        if (h_k % dilation_h == 0 && w_k % dilation_w == 0) {
          h_k /= dilation_h;
          w_k /= dilation_w;
          int data_col_index = (((c_im * kernel_h + h_k) * kernel_w + w_k) *
                                height_col + h_col) * width_col + w_col;
          val = static_cast<float>(val) + static_cast<float>(data_col[data_col_index]);
        }
      }
    }
    data_im[index] = val;
  }
}


template <typename Dtype>
void col2im_gpu(const Dtype* data_col, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w, const int stride_h,
    const int stride_w, const int dilation_h, const int dilation_w,
    Dtype* data_im) {
  int height_col = (height + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1)) /
      stride_h + 1;
  int width_col = (width + 2 * pad_w - (dilation_w * (kernel_w - 1) + 1)) /
      stride_w + 1;
  int num_kernels = channels * height * width;
  // To avoid involving atomic operations, we will launch one kernel per
  // bottom dimension, and then in the kernel add up the top dimensions.
  // NOLINT_NEXT_LINE(whitespace/operators)
  col2im_gpu_kernel<Dtype><<<CAFFE_GET_BLOCKS(num_kernels),
                             CAFFE_CUDA_NUM_THREADS>>>(
      num_kernels, data_col, height, width, channels, kernel_h, kernel_w,
      pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w,
      height_col, width_col, data_im);
  CUDA_POST_KERNEL_CHECK;
}

// Explicit instantiation
template void col2im_gpu<float>(const float* data_col, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w, const int stride_h,
    const int stride_w, const int dilation_h, const int dilation_w,
    float* data_im);
template void col2im_gpu<double>(const double* data_col, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w, const int stride_h,
    const int stride_w, const int dilation_h, const int dilation_w,
    double* data_im);
template void col2im_gpu<__half>(const __half* data_col, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w, const int stride_h,
    const int stride_w, const int dilation_h, const int dilation_w,
    __half* data_im);


int main() {
    Tensor4d *data = new Tensor4d(2, 3, 7, 7);
    Tensor4d *kernel = new Tensor4d(2, 3, 3, 3);
    data->generate_rand(8);
    kernel->generate_rand(3);
    cout << "data:" << endl;
    data->print_tensor();
    ConvParam *param = new ConvParam(data, kernel, 1, 1);

    MatrixParam *m = param->im2col();
    float *c = new float[m->m * m->k];
    for (int i = 0; i < m->m * m->k; i++) {
        c[i] = __half2float(m->A[i]);
    }

    float *in = new float[data->n*data->c*data->h*data->w];
    for (int i = 0; i < data->n*data->c*data->h*data->w;i++) {
        in[i] = __half2float(data->tensor[i]);
        //printf("%f ", in[i]);
    }
    printf("\n");
    float *d_in;
    cudaMalloc((void**) &d_in, data->n*data->c*data->h*data->w*sizeof(float));
    cudaMemcpy(d_in, in, data->n*data->c*data->h*data->w*sizeof(float), cudaMemcpyHostToDevice);

    float *d_out;
    cudaMalloc((void**) &d_out, data->n * param->getOut_height() * param->getOut_width() * kernel->c * kernel->h * kernel->w*sizeof(float));
    float *out = new float[data->n * param->getOut_height() * param->getOut_width() * kernel->c * kernel->h * kernel->w];

    im2col_gpu<float>(d_in, data->n, data->c, data->h, data->w, kernel->h, kernel->w, 1, 1, 1, 1, 1, 1, d_out);

    cudaMemcpy(out, d_out, data->n * param->getOut_height() * param->getOut_width() * kernel->c * kernel->h * kernel->w*sizeof(float), cudaMemcpyDeviceToHost);

    printf("im2col out:\n");
    for (int i = 0; i < kernel->c * kernel->h * kernel->w; i++) {
        for (int j = 0; j < data->n * param->getOut_height() * param->getOut_width(); j++) {
            printf("%d ", int(out[i * data->n * param->getOut_height() * param->getOut_width() + j]));
        }
        printf("\n");
    }

    printf("total: %d | %d", m->m * m->k, data->n * param->getOut_height() * param->getOut_width() * kernel->c * kernel->h * kernel->w);
    int cnt = 0;
    for (int i = 0; i < m->m; i++) {
        for (int j = 0; j < m->k; j++) {
            if (c[i * m->k + j] != out[j * m->m + i]) {
                cnt++;
                 printf("%d : %d ", int(c[i * m->k + j]), int(out[j * m->m + i]));
            }
        }
    }
    printf("\ndiff: %d", cnt);



//     MatrixParam *m = param->im2col();
//     m->matmul_cpu();
//     m->print_all();
//     printf("==%d==", param->getOut_height());
// //     printf("D:\n");
// //     m->print_matrix(m->D, m->m, m->n);
//
//     float *in_rev = new float[m->m * m->n];
//     for (int i = 0; i < m->m * m->n;i++) {
//         in_rev[i] = __half2float(m->D[i]);
//     }
//     float *d_in_rev;
//     cudaMalloc((void**) &d_in_rev, m->m * m->n * sizeof(float));
//     cudaMemcpy(d_in_rev, in_rev,  m->m * m->n * sizeof(float), cudaMemcpyHostToDevice);
//
//     float *d_rev;
//     //cudaMalloc((void**) &d_rev, data->n * param->getOut_height() * param->getOut_width() * kernel->n * sizeof(float));
//     cudaMalloc((void**) &d_rev, 25 * sizeof(float));
//
//     col2im_gpu(d_in_rev, data->c, data->h, data->w, kernel->h, kernel->w, 0, 0, 1, 1, 1, 1, d_rev);
//
//     float *rev = new float[25];
//     cudaMemcpy(rev, d_rev, 25 * sizeof(float), cudaMemcpyDeviceToHost);
//
//     printf("col2im out:\n");
//     for (int i = 0; i < 25; i++)  printf("%f ", rev[i]);
//     return 0;
//
//     for (int i = 0; i < data->n; i++) {
//         for (int j = 0; j < kernel->n; j++) {
//             for (int k = 0; k < param->getOut_height(); k++) {
//                 for (int v = 0; v < param->getOut_width(); v++) {
//                     printf("%f ", rev[i * kernel->n * param->getOut_height() * param->getOut_width() + j * param->getOut_height() * param->getOut_width()
//                         + k * param->getOut_width() + v]);
//                 }
//                 printf("\n");
//             }
//             printf("\n");
//         }
//         printf("\n");
//     }
}