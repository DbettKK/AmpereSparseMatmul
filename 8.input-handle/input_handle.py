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


def im2col(data: np.ndarray, core: np.ndarray, stride: int, padding: int) -> (np.ndarray, np.ndarray):
    # 得到转换后的data矩阵和core矩阵
    # 代码后期补
    # output_h = (data_h + 2 * padding - core_h) // stride + 1
    # output_w = (data_w + 2 * padding - core_w) // stride + 1
    return data, core


def pad_zero(item: np.ndarray, pad_row: bool, pad_num: int):
    # 根据输入决定padding的方向和数目
    # 默认都是向下or右
    pass


def handle_input(data: np.ndarray, core: np.ndarray, stride: int, padding: int) -> (np.ndarray, np.ndarray):
    # 处理输入的矩阵 进行填充使得其满足spmma指令的需要
    trans_data, trans_core = im2col(data, core, stride, padding)

    a_h, a_w = trans_data.shape
    b_h, b_w = trans_core.shape

    if a_h % 8 != 0:
        pad_zero(trans_data, True, 8 - a_h % 8)
    if a_w % 8 != 0:
        pad_zero(trans_data, False, 8 - a_w % 8)
    if b_h % 8 != 0:
        pad_zero(trans_core, True, 8 - b_h % 8)
    if b_w % 8 != 0:
        pad_zero(trans_core, False, 8 - b_w % 8)

    return trans_data, trans_core


if __name__ == '__main__':
    input()
