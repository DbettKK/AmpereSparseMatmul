import time

import numpy as np
import torch


def im2col(input_data: np.ndarray, filter_h, filter_w, stride=1, pad=0) -> np.ndarray:
    """
    Parameters
    ----------
    input_data : 由(数据量, 通道, 高, 长)的4维数组构成的输入数据
    filter_h : 滤波器的高
    filter_w : 滤波器的长
    stride : 步幅
    pad : 填充

    Returns
    -------
    col : 2维数组
    """
    N, C, H, W = input_data.shape
    out_h = (H + 2 * pad - filter_h) // stride + 1  # 输出矩阵的高
    out_w = (W + 2 * pad - filter_w) // stride + 1  # 输出矩阵的宽

    img = np.pad(input_data, [(0, 0), (0, 0), (pad, pad), (pad, pad)], 'constant')
    col = np.zeros((N, C, filter_h, filter_w, out_h, out_w))

    for y in range(filter_h):
        y_max = y + stride * out_h
        for x in range(filter_w):
            x_max = x + stride * out_w
            col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]

    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N * out_h * out_w, -1)
    return col


def torch_conv(data, core):
    data = torch.from_numpy(data)
    core = torch.from_numpy(core.reshape(1, 3, 2, 2))
    start = time.time()
    print(torch.nn.functional.conv2d(input=data, weight=core))
    print(time.time() - start)


def im2col_conv(data, core):

    data = im2col(data, 2, 2)
    core = core.reshape(12, 1)
    start = time.time()
    print(np.matmul(data, core).reshape(3, 3))
    print(time.time() - start)


def main():
    input_data = np.arange(16*3, dtype=int).reshape(1, 3, 4, 4)
    input_core = np.ones(12, dtype=int)

    torch_conv(input_data, input_core)
    im2col_conv(input_data, input_core)

    #data = im2col(input_data, 2, 2)

    # data1 = data.T.reshape(3, 4, 9).transpose(0, 2, 1)
    # core1 = np.ones(12).reshape(3, 4, 1)
    # print(np.einsum("ijk,ikn->ijn", data1, core1).reshape(3, 3, 3))

    #data2 = data
    #core2 = np.ones(12).reshape(12, 1)
    #print(np.matmul(data2, core2).reshape(3, 3))


if __name__ == '__main__':
    main()
