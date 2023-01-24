import pytest
import numpy as np
import scipy.ndimage
import torch

from whisper.timing import dtw_cpu, dtw_cuda, median_filter


sizes = [
    (10, 20), (32, 16), (123, 1500), (234, 189),
]
shapes = [
    (4, 5, 20, 345), (6, 12, 240, 512),
]


@pytest.mark.parametrize("N, M", sizes)
def test_dtw(N: int, M: int):
    steps = np.concatenate([np.zeros(N - 1), np.ones(M - 1)])
    np.random.shuffle(steps)
    x = np.random.random((N, M)).astype(np.float32)

    i, j, k = 0, 0, 0
    trace = []
    while True:
        x[i, j] -= 1
        trace.append((i, j))

        if k == len(steps):
            break

        if k + 1 < len(steps) and steps[k] != steps[k + 1]:
            i += 1
            j += 1
            k += 2
            continue

        if steps[k] == 0:
            i += 1
        if steps[k] == 1:
            j += 1
        k += 1

    trace = np.array(trace).T
    dtw_trace = dtw_cpu(x)

    assert np.allclose(trace, dtw_trace)


@pytest.mark.requires_cuda
@pytest.mark.parametrize("N, M", sizes)
def test_dtw_cuda_equivalence(N: int, M: int):
    x_numpy = np.random.randn(N, M).astype(np.float32)
    x_cuda = torch.from_numpy(x_numpy).cuda()

    trace_cpu = dtw_cpu(x_numpy)
    trace_cuda = dtw_cuda(x_cuda)

    assert np.allclose(trace_cpu, trace_cuda)


@pytest.mark.parametrize("shape", shapes)
def test_median_filter(shape):
    x = torch.randn(*shape)

    for filter_width in [3, 5, 7, 13]:
        filtered = median_filter(x, filter_width)
        scipy_filtered = scipy.ndimage.median_filter(x, (1, 1, 1, filter_width), mode="nearest")

        assert np.allclose(filtered, scipy_filtered)


@pytest.mark.requires_cuda
@pytest.mark.parametrize("shape", shapes)
def test_median_filter_equivalence(shape):
    x = torch.randn(*shape)

    for filter_width in [3, 5, 7, 13]:
        filtered_cpu = median_filter(x, filter_width)
        filtered_gpu = median_filter(x.cuda(), filter_width).cpu()

        assert np.allclose(filtered_cpu, filtered_gpu)
