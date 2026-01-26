from functools import lru_cache

import numpy as np
import torch

try:
    import triton
    import triton.language as tl
except ImportError:
    raise RuntimeError("triton import failed; try `pip install --pre triton`")


@triton.jit
def dtw_kernel(
    cost, trace, x, x_stride, cost_stride, trace_stride, N, M, BLOCK_SIZE: tl.constexpr
):
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
        tl.store(trace_ptr + offsets, 2, mask=mask & (c2 <= c0) & (c2 <= c1))
        tl.store(trace_ptr + offsets, 1, mask=mask & (c1 <= c0) & (c1 <= c2))
        tl.store(trace_ptr + offsets, 0, mask=mask & (c0 <= c1) & (c0 <= c2))


@lru_cache(maxsize=None)
def median_kernel(filter_width: int):
    @triton.jit
    def kernel(
        y, x, x_stride, y_stride, BLOCK_SIZE: tl.constexpr
    ):  # x.shape[-1] == filter_width
        row_idx = tl.program_id(0)
        offsets = tl.arange(0, BLOCK_SIZE)
        mask = offsets < y_stride

        x_ptr = x + row_idx * x_stride  # noqa: F841
        y_ptr = y + row_idx * y_stride

        LOAD_ALL_ROWS_HERE  # noqa: F821

        BUBBLESORT_HERE  # noqa: F821

        tl.store(y_ptr + offsets, MIDDLE_ROW_HERE, mask=mask)  # noqa: F821

    kernel = triton.JITFunction(kernel.fn)
    new_kernel = kernel.src.replace(
        "    LOAD_ALL_ROWS_HERE",
        "\n".join(
            [
                f"    row{i} = tl.load(x_ptr + offsets + {i}, mask=mask)"
                for i in range(filter_width)
            ]
        ),
    )

    new_kernel = new_kernel.replace(
        "    BUBBLESORT_HERE",
        "\n\n".join(
            [
                "\n\n".join(
                    [
                        "\n".join(
                            [
                                f"    smaller = tl.where(row{j} < row{j + 1}, row{j}, row{j + 1})",
                                f"    larger = tl.where(row{j} > row{j + 1}, row{j}, row{j + 1})",
                                f"    row{j} = smaller",
                                f"    row{j + 1} = larger",
                            ]
                        )
                        for j in range(filter_width - i - 1)
                    ]
                )
                for i in range(filter_width // 2 + 1)
            ]
        ),
    )

    new_kernel = new_kernel.replace("MIDDLE_ROW_HERE", f"row{filter_width // 2}")

    if hasattr(kernel, "_unsafe_update_src") is True:
        kernel._unsafe_update_src(new_kernel)
        kernel.hash = None
    else:
        kernel.src = new_kernel

    return kernel


def median_filter_cuda(x: torch.Tensor, filter_width: int):
    """Apply a median filter of given width along the last dimension of x"""
    slices = x.contiguous().unfold(-1, filter_width, 1)
    grid = np.prod(slices.shape[:-2])

    kernel = median_kernel(filter_width)
    y = torch.empty_like(slices[..., 0])

    BLOCK_SIZE = 1 << (y.stride(-2) - 1).bit_length()
    kernel[(grid,)](y, x, x.stride(-2), y.stride(-2), BLOCK_SIZE=BLOCK_SIZE)

    return y
