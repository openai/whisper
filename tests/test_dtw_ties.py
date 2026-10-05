"""Exact alignment checks for tied dynamic-programming predecessors."""

import itertools

import numpy as np
import pytest
import torch

from whisper.timing import dtw_cpu, dtw_cuda


def reference_path(values):
    """Independent recurrence with diagonal, vertical, horizontal tie priority."""
    rows, columns = values.shape
    costs = np.full((rows + 1, columns + 1), np.inf, dtype=np.float64)
    costs[0, 0] = 0
    parents = {}
    for row in range(1, rows + 1):
        for column in range(1, columns + 1):
            choices = [(row - 1, column - 1), (row - 1, column), (row, column - 1)]
            parent = min(choices, key=lambda location: costs[location])
            costs[row, column] = float(values[row - 1, column - 1]) + costs[parent]
            parents[row, column] = parent
    path = []
    location = rows, columns
    while location != (0, 0):
        row, column = location
        assert row > 0 and column > 0
        path.append((row - 1, column - 1))
        location = parents[location]
    return np.array(path[::-1]).T, costs[-1, -1]


CASES = [
    np.array([[0, 0], [1, 0]]),
    np.array([[0, 1], [0, 0]]),
    np.zeros((3, 5)),
    np.zeros((5, 3)),
    np.array([[0, -1, 0], [-1, 0, -1], [0, -1, 0]]),
    np.array([[1, 0, 0, 2], [0, 0, 1, 0], [2, 1, 0, 0]]),
]


@pytest.mark.parametrize("values", CASES)
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_tied_predecessors_match_the_minimum_cost_path(values, dtype):
    values = values.astype(dtype)
    expected, cost = reference_path(values)
    actual = dtw_cpu(values)
    np.testing.assert_array_equal(actual, expected)
    assert float(values[tuple(actual)].sum()) == cost


def test_all_binary_three_by_three_cost_matrices():
    for entries in itertools.product([0, 1], repeat=9):
        values = np.array(entries, dtype=np.float32).reshape(3, 3)
        expected, cost = reference_path(values)
        actual = dtw_cpu(values)
        np.testing.assert_array_equal(actual, expected, err_msg=str(values))
        assert float(values[tuple(actual)].sum()) == cost


@pytest.mark.requires_cuda
@pytest.mark.parametrize("values", CASES)
def test_tied_cpu_paths_agree_with_cuda(values):
    values = values.astype(np.float32)
    actual = dtw_cuda(torch.from_numpy(values).cuda())
    np.testing.assert_array_equal(actual, dtw_cpu(values))
