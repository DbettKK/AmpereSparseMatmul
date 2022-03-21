import tvm
from tvm import te


if __name__ == '__main__':
    n = te.var("n")
    A = te.placeholder((n,), name='A')
    # 这行替换为tvm.exp
    B = te.compute(A.shape, lambda i: te.exp(A[i]), name="B")
    s = te.create_schedule(B.op)
    num_thread = 64
    bx, tx = s[B].split(B.op.axis[0], factor=num_thread)
    s[B].bind(bx, te.thread_axis("blockIdx.x"))
    s[B].bind(tx, te.thread_axis("threadIdx.x"))
    fcuda = tvm.build(s, [A, B], "cuda", name="myexp")
    print(fcuda.imported_modules[0].get_source())
