try:
    import triton
    import triton.language as tl
except ImportError:
    raise RuntimeError("triton import failed; try `pip install triton`")


@triton.jit
def dtw_kernel(cost, trace, x, x_stride, cost_stride, trace_stride, N, M, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < M

    for k in range(1, N + M + 1):  # k = i + j
        tl.debug_barrier()

        p0 = cost + (k - 1) * cost_stride
        p1 = cost + k * cost_stride
        p2 = cost + k * cost_stride + 1

        c0 = tl.load(p0 + offsets, mask=mask)
        c1 = tl.load(p1 + offsets, mask=mask)
        c2 = tl.load(p2 + offsets, mask=mask)

        x_row = tl.load(x + (k - 1) * x_stride + offsets, mask=mask, other=0)
        cost_row = x_row + tl.minimum(tl.minimum(c0, c1), c2)

        cost_ptr = cost + (k + 1) * cost_stride + 1
        tl.store(cost_ptr + offsets, cost_row, mask=mask)

        trace_ptr = trace + (k + 1) * trace_stride + 1
        tl.store(trace_ptr + offsets, 0, mask=mask & (c0 <= c1) & (c0 <= c2))
        tl.store(trace_ptr + offsets, 1, mask=mask & (c1 <= c0) & (c1 <= c2))
        tl.store(trace_ptr + offsets, 2, mask=mask & (c2 <= c0) & (c2 <= c1))
