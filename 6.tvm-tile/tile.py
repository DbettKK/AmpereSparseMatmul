import tvm
from tvm import te
import numpy as np


M = 12544
N = 1
K = 49


def im2col(input_data: np.ndarray, filter_h, filter_w, stride=1, pad=0) -> np.ndarray:
    """
    ResNet50
    224 * 224   7 * 7
    padding -> 230 * 230
    stride -> 2
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


def get_target() -> (tvm.target.target.Target, tvm.runtime.ndarray):
    target = tvm.target.Target(target="llvm", host="llvm")
    device = tvm.device(target.kind.name, 0)
    return target, device


def placeholder(data, core):
    k = te.reduce_axis((0, K), "k")
    a = te.placeholder((M, N), dtype="float32", name="A")
    b = te.placeholder((N, K), dtype="float32", name="B")
    c = te.compute((M, N), lambda x, y: te.sum(a[x, k] * b[k, y], axis=k), name="C")
    s = te.create_schedule(c.op)
    tgt, device = get_target()
    func = tvm.build(s, [a, b, c], target=tgt, name="conv")
    run_data = tvm.nd.array(data.astype(dtype="float32"), device)
    run_core = tvm.nd.array(core.astype(dtype="float32"), device)
    run_out = tvm.nd.array(np.zeros((M, N), dtype="float32"), device)
    func(run_data, run_core, run_out)


def main():
    pass


if __name__ == "__main__":
    main()



