import numpy as np


def input(data: np.ndarray, core: np.ndarray, stride: int, padding: int):
    """

    Parameters
    ----------
    data: 数据矩阵 2d W * H
    core: 卷积核 3d 包含channel C * W * H
    stride: 步长
    padding:

    Returns
    -------

    """
    handle_input()


def handle_input():
    # 处理输入的矩阵 进行填充使得其满足spmma指令的需要
    pass


if __name__ == '__main__':
    input()
