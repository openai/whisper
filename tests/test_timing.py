import pytest
import numpy as np
import torch

from whisper.timing import dtw_cpu, dtw_cuda


sizes = [
    (10, 20), (32, 16), (123, 1500), (234, 189)
]


@pytest.mark.parametrize("N, M", sizes)
def test_dtw(N: int, M: int):
    np.random.seed(42)
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
    np.random.seed(42)
    x_numpy = np.random.randn(N, M).astype(np.float32)
    x_cuda = torch.from_numpy(x_numpy).cuda()

    trace_cpu = dtw_cpu(x_numpy)
    trace_cuda = dtw_cuda(x_cuda)

    assert np.allclose(trace_cpu, trace_cuda)
