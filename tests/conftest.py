import random as rand

import numpy
import pytest
import torch


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "requires_cuda: mark test as requiring a CUDA GPU"
    )


def pytest_collection_modifyitems(config, items):
    if not torch.cuda.is_available():
        skip_cuda = pytest.mark.skip(reason="CUDA not available")
        for item in items:
            if item.get_closest_marker("requires_cuda"):
                item.add_marker(skip_cuda)


@pytest.fixture
def random():
    rand.seed(42)
    numpy.random.seed(42)
