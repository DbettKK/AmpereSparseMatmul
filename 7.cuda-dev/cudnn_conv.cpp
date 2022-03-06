#include<cstdio>
#include<cstdlib>
#include<cstdint>
#include<cstring>

#include<fstream>
#include<iostream>

#include<cudnn.h>

#include"cuda_utils.h"

using namespace std;




int main() {
    // read from bin_file
    int data_n = 1, data_c = 1, data_w = 5, data_h = 5;
    int data_size = data_n * data_c * data_w * data_h * sizeof(float);
    int kernel_n = 12, kernel_c = 1, kernel_w = 4, kernel_h = 4;
    int kernel_size = kernel_n * kernel_c * kernel_w * kernel_h * sizeof(float);

    float **files = read_bin(data_size, kernel_size);
    //handle
    cudnnHandle_t handle;
    CHECK_CUDNN(cudnnCreate(&handle))

    // input
    float *input = files[0];
    cout << "input: " << endl;
    print_tensor(input, data_n, data_c, data_w, data_h);

    cudnnTensorDescriptor_t input_descriptor;
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&input_descriptor))
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(input_descriptor,
                               CUDNN_TENSOR_NHWC,
                               CUDNN_DATA_FLOAT,
                               data_n, data_c, data_w, data_h)) // n, c, w, h


    // kernel
    float *kernel = files[1];
    cout << "kernel: " << endl;
    print_tensor(kernel, kernel_n, kernel_c, kernel_w, kernel_h);
    cudnnFilterDescriptor_t kernel_descriptor;
    CHECK_CUDNN(cudnnCreateFilterDescriptor(&kernel_descriptor))
    CHECK_CUDNN(cudnnSetFilter4dDescriptor(kernel_descriptor,
                               CUDNN_DATA_FLOAT,
                               CUDNN_TENSOR_NCHW,
                               kernel_n, kernel_c, kernel_w, kernel_h))


    // convolution descriptor
    cudnnConvolutionDescriptor_t conv_descriptor;
    CHECK_CUDNN(cudnnCreateConvolutionDescriptor(&conv_descriptor))
    CHECK_CUDNN(cudnnSetConvolution2dDescriptor(conv_descriptor,
                                    1, 1, // zero-padding
                                    1, 1, // stride
                                    1, 1, // dilation 卷积核膨胀 膨胀后用0填充空位
                                    // 卷积是需要将卷积核旋转180°再进行后续的 -> CUDNN_CONVOLUTION
                                    CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT))


    // output
    int out_n, out_c, out_h, out_w;
    CHECK_CUDNN(cudnnGetConvolution2dForwardOutputDim(conv_descriptor,
                               input_descriptor,
                               kernel_descriptor,
                               &out_n, &out_c, &out_h, &out_w))
    printf("output: %d * %d * %d * %d\n", out_n, out_c, out_h, out_w);
    int out_size = out_n * out_c * out_h * out_w * sizeof(float);
    float *output = (float *)malloc(out_size);
    cudnnTensorDescriptor_t output_descriptor;
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&output_descriptor))
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(output_descriptor,
                               CUDNN_TENSOR_NHWC,
                               CUDNN_DATA_FLOAT,
                               out_n, out_c, out_h, out_w))


    // algorithm
    cudnnConvolutionFwdAlgoPerf_t algo_perf[4];
    int ret;
    CHECK_CUDNN(cudnnFindConvolutionForwardAlgorithm(handle,
                                        input_descriptor,
                                        kernel_descriptor,
                                        conv_descriptor,
                                        output_descriptor,
                                        4,
                                        &ret,
                                        algo_perf))

    cudnnConvolutionFwdAlgo_t algo;
    bool flag = false;
    for (int i = 0; i < ret; i++) {
        if (algo_perf[i].status == CUDNN_STATUS_SUCCESS) {
            algo = algo_perf[i].algo;
            flag = true;
            break;
        }
    }
    if (!flag) {
        cout << "no alg" << endl;
        return 0;
    }

    // workspace size && allocate memory
    size_t workspace_size = 0;
    CHECK_CUDNN( cudnnGetConvolutionForwardWorkspaceSize(handle,
                                            input_descriptor,
                                            kernel_descriptor,
                                            conv_descriptor,
                                            output_descriptor,
                                            algo,
                                            &workspace_size) )

    void * workspace = nullptr;
    cudaMalloc(&workspace, workspace_size);

    // convolution
    auto alpha = 1.0f, beta = 0.0f;

    float *d_input, *d_kernel, *d_output;

    CHECK_CUDA( cudaMalloc((void**) &d_input, data_size) )
    CHECK_CUDA( cudaMalloc((void**) &d_kernel, kernel_size) )
    CHECK_CUDA( cudaMalloc((void**) &d_output, out_size) )

    CHECK_CUDA( cudaMemcpy(d_input, input, data_size, cudaMemcpyHostToDevice) )
    CHECK_CUDA( cudaMemcpy(d_kernel, kernel, kernel_size, cudaMemcpyHostToDevice) )


    // calculate
    CHECK_CUDNN( cudnnConvolutionForward(handle,
                            &alpha, input_descriptor, d_input,
                            kernel_descriptor, d_kernel,
                            conv_descriptor, algo,
                            workspace, workspace_size,
                            &beta, output_descriptor, d_output) )



    cudaMemcpy(output, d_output, out_size, cudaMemcpyDeviceToHost);

    // destroy
    cudaFree(workspace);

    CHECK_CUDNN(cudnnDestroyTensorDescriptor(input_descriptor))
    CHECK_CUDNN(cudnnDestroyTensorDescriptor(output_descriptor))
    CHECK_CUDNN(cudnnDestroyConvolutionDescriptor(conv_descriptor))
    CHECK_CUDNN(cudnnDestroyFilterDescriptor(kernel_descriptor))

    CHECK_CUDNN(cudnnDestroy(handle))

    cout << "output: " << endl;
    print_tensor(output, out_n, out_c, out_w, out_h);

    return 0;
}