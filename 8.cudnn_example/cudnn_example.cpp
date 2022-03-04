#include<cudnn.h>
#include<iostream>

using namespace std;

#define CHECK_CUDNN(func)                                                       \
{                                                                              \
    cudnnStatus_t status = (func);                                               \
    if (status != CUDNN_STATUS_SUCCESS) {                                               \
        printf("CUDNN failed at line %d with error: %s (%d)\n",             \
               __LINE__, cudnnGetErrorString(status), status);                  \
        return 0;                                                   \
    }                                                                          \
}

void print_tensor(float *item, int n, int c, int w, int h) {
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


int main() {
    //handle
    cudnnHandle_t handle;
    cudnnCreate(&handle);

    // input
    float *input = new float[1 * 1 * 16 * 16];
    float *output = new float[1 * 1 * 14 * 14];
    for (int i = 0; i < 256; i++) {
        input[i] = rand() % 8;
    }
    cout << "input: " << endl;
    print_tensor(input, 1, 1, 16, 16);
    // cudnn
    cudnnTensorDescriptor_t input_descriptor;
    cudnnCreateTensorDescriptor(&input_descriptor);
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(input_descriptor,
                               CUDNN_TENSOR_NHWC,
                               CUDNN_DATA_FLOAT,
                               1, 1, 16, 16)) // n, c, w, h <int>
                               //input.shape(0), input.shape(1), input.shape(2), input.shape(3));

    // output
    cudnnTensorDescriptor_t output_descriptor;
    cudnnCreateTensorDescriptor(&output_descriptor);
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(output_descriptor,
                               CUDNN_TENSOR_NHWC,
                               CUDNN_DATA_FLOAT,
                               1, 1, 14, 14))
                               // output.shape(0), output.shape(1), output.shape(2), output.shape(3));

    // kernel <12, 1, 3, 3>
    float *kernel = new float[1 * 1 * 3 * 3];
    for (int i = 0; i < 1 * 9; i++) {
        if (i % 2) kernel[i] = 0;
        else kernel[i] = rand() % 5;
    }
    cout << "kernel: " << endl;
    print_tensor(kernel, 1, 1, 3, 3);
    cudnnFilterDescriptor_t kernel_descriptor;
    cudnnCreateFilterDescriptor(&kernel_descriptor);
    CHECK_CUDNN( cudnnSetFilter4dDescriptor(kernel_descriptor,
                               CUDNN_DATA_FLOAT,
                               CUDNN_TENSOR_NCHW,
                               1, 1, 3, 3))
                               //kernel.shape(0), kernel.shape(1), kernel.shape(2), kernel.shape(3));
    // convolution descriptor
    cudnnConvolutionDescriptor_t conv_descriptor;
    cudnnCreateConvolutionDescriptor(&conv_descriptor);
    CHECK_CUDNN(cudnnSetConvolution2dDescriptor(conv_descriptor,
                                    1, 1, // zero-padding
                                    1, 1, // stride
                                    1, 1, // dilation 卷积核膨胀 膨胀后用0填充空位
                                    // 卷积是需要将卷积核旋转180°再进行后续的 -> CUDNN_CONVOLUTION
                                    CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT))

    // algorithm
    cudnnConvolutionFwdAlgoPerf_t algo_perf[8];
    int ret;
    CHECK_CUDNN(cudnnFindConvolutionForwardAlgorithm(handle,
                                        input_descriptor,
                                        kernel_descriptor,
                                        conv_descriptor,
                                        output_descriptor,
                                        8,
                                        &ret,
                                        algo_perf))

    cout << ret << endl;
    cudnnConvolutionFwdAlgo_t algo;
    bool flag = false;
    for (int i = 0; i < ret; i++) {
        cout << algo_perf[i].status << ", ";
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
    cudnnGetConvolutionForwardWorkspaceSize(handle,
                                            input_descriptor,
                                            kernel_descriptor,
                                            conv_descriptor,
                                            output_descriptor,
                                            algo,
                                            &workspace_size);

    void * workspace = nullptr;
    cudaMalloc(&workspace, workspace_size);

    // convolution
    auto alpha = 1.0f, beta = 0.0f;
    cudnnConvolutionForward(handle,
                            &alpha, input_descriptor, input,
                            kernel_descriptor, kernel,
                            conv_descriptor, algo,
                            workspace, workspace_size,
                            &beta, output_descriptor, output);


    float *out = new float[12 * 14 * 14];
    cudaMemcpy(out, output, 12 * 14 * 14 * sizeof(float), cudaMemcpyDeviceToHost);

    // destroy
    cudaFree(workspace);

    cudnnDestroyTensorDescriptor(input_descriptor);
    cudnnDestroyTensorDescriptor(output_descriptor);
    cudnnDestroyConvolutionDescriptor(conv_descriptor);
    cudnnDestroyFilterDescriptor(kernel_descriptor);

    cudnnDestroy(handle);

    cout << "output: " << endl;
    //print_tensor(out, 1, 12, 14, 14);

    return 0;
}