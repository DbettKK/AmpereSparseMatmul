import random

import numpy as np


def make_dense_mat(row, col, dtype):
    return np.random.rand(row, col).astype(dtype)


def make_zero_mat(row, col, dtype):
    return np.zeros(row * col).reshape(row, col).astype(dtype)


def make_sparse_mat(row, col, dtype):
    a = np.random.rand(row, col).astype(dtype)
    for i in range(row):
        for j in range(col // 4):
            i1 = random.randint(0, 3)
            i2 = random.randint(0, 3)
            while i2 == i1:
                i2 = random.randint(0, 3)
            a[i, j * 4 + i1] = 0
            a[i, j * 4 + i2] = 0
        # 防止像k = 11这种情况时导致A矩阵不够sparse
        if col % 4 > 1:
            i1 = random.randint(0, col % 4 - 1)
            a[i, col - col % 4 + i1] = 0
    return a


def make_tensor(n, c, w, h, _dtype):
    return np.random.randint(0, 5, (n, c, w, h)).astype(_dtype)


def im2col(input_data: np.ndarray, filter_h, filter_w, stride=1, pad=0):
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


def kernel_im2col(core, core_n, core_c, core_w, core_h):
    return core.reshape(core_n, core_c * core_w * core_h).T


if __name__ == '__main__':
    data_n, data_c, data_w, data_h = 1, 1, 5, 5
    kernel_n, kernel_c, kernel_w, kernel_h = 12, 1, 4, 4
    padding = 1
    stride = 1
    dtype = 'float32'

    data = make_tensor(data_n, data_c, data_w, data_h, dtype)
    kernel = make_tensor(kernel_n, kernel_c, kernel_w, kernel_h, dtype)

    data_trans = im2col(data, kernel_h, kernel_w, stride, padding)
    data_trans_w, data_trans_h = data_trans.shape
    kernel_trans = kernel_im2col(kernel, kernel_n, kernel_c, kernel_w, kernel_h)

    m = data_trans_w
    k = data_trans_h
    n = kernel_n
    print(data)
    print(kernel)
    print(m, k, n)
    print(data_trans)
    print(kernel_trans)

    zero_matrix = make_zero_mat(m, n, dtype)

    # to bin_file
    data.tofile('data.bin')
    kernel.tofile('kernel.bin')
    data_trans.tofile('a.bin')
    kernel_trans.tofile('b.bin')
    zero_matrix.tofile('c.bin')

    print(np.matmul(data_trans, kernel_trans))
