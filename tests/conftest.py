import random as rand

import numpy
import pytest


def pytest_configure(config):
    config.addinivalue_line("markers", "requires_cuda")
    config.addinivalue_line("markers", "requires_hpu")


@pytest.fixture
def random():
    rand.seed(42)
    numpy.random.seed(42)
