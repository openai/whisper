import random as rand

import numpy
import pytest


@pytest.fixture
def random():
    rand.seed(42)
    numpy.random.seed(42)
