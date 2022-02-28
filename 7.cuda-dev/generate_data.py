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
    return a


if __name__ == '__main__':
    m = 16
    k = 16
    n = 8
    mat_a = make_sparse_mat(m, k, 'float16')
    mat_b = make_dense_mat(k, n, 'float16')
    mat_c = make_zero_mat(m, n, 'float16')
    mat_a.tofile('a.bin')
    mat_b.tofile('b.bin')
    mat_c.tofile('c.bin')
    # print(mat_a)
    # print(mat_b)
    # print(mat_c)
