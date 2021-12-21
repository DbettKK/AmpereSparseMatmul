import tvm
from tvm import te
import numpy as np


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


def generate_data(data_shape: list, core_shape: list) -> (np.ndarray, np.ndarray):
    data_h, data_w = data_shape
    core_h, core_w = core_shape
    data = np.arange(data_h * data_w, dtype=int).reshape(data_h, data_w)
    core = np.ones(core_h * core_w, dtype=int).reshape(core_h, core_w)
    return data, core


def im2col_self(data: np.ndarray, core: np.ndarray) -> (np.ndarray, np.ndarray):
    input_data = data.reshape(1, 1, data.shape[0], data.shape[1])
    out_size = data.shape[0] - core.shape[0] + 1
    transfer_data = im2col(input_data, core.shape[0], core.shape[1])
    transfer_core = core.reshape(core.shape[0]*core.shape[1], 1)
    return transfer_data, transfer_core


def get_target() -> (tvm.target.target.Target, tvm.runtime.ndarray):
    target = tvm.target.Target(target="llvm", host="llvm")
    device = tvm.device(target.kind.name, 0)
    return target, device


def tensor_expr(data: np.ndarray, core: np.ndarray, target: tvm.target.target.Target, device: tvm.runtime.ndarray):
    if data.shape[1] != core.shape[0]:
        print("shape不匹配")
        exit(0)
    M, K, N = data.shape[0], data.shape[1], core.shape[1]
    k = te.reduce_axis((0, K), "k")
    Data = te.placeholder((M, K), name="Data")
    Core = te.placeholder((K, N), name="Core")
    Out = te.compute((M, N), lambda x, y: te.sum(Data[x, k] * Core[k, y], axis=k), name="Out")
    # 调度
    sche = te.create_schedule(Out.op)
    func_conv = tvm.build(sche, [Data, Core, Out], target=target, name="conv")
    run_data = tvm.nd.array(data.astype(dtype="float32"), device)
    run_core = tvm.nd.array(core.astype(dtype="float32"), device)
    run_out = tvm.nd.array(np.zeros((M, N), dtype="float32"), device)
    func_conv(run_data, run_core, run_out)
    print(run_out.numpy().reshape(4, 4))
    print(func_conv.get_source())


def main():
    data, core = generate_data([5, 5], [2, 2])
    trans_data, trans_core = im2col_self(data, core)
    tat, dev = get_target()
    tensor_expr(trans_data, trans_core, tat, dev)


if __name__ == "__main__":
    main()
