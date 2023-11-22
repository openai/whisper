import pytest
from typing import Optional

from whisper.utils import optional_float, optional_int, str2bool


@pytest.mark.parametrize(("provided", "expected"), [
    ("TRUE", True),
    ("True", True),
    ("true", True),
    ("YES", True),
    ("Yes", True),
    ("yes", True),
    ("Y", True),
    ("y", True),
    ("1", True),

    ("FALSE", False),
    ("False", False),
    ("false", False),
    ("NO", False),
    ("No", False),
    ("no", False),
    ("N", False),
    ("n", False),
    ("0", False),
])
def test_str2bool(provided: str, expected: bool) -> None:
    assert str2bool(provided) is expected


def test_str2bool_faulty_argument() -> None:
    with pytest.raises(ValueError, match="Expected one of"):
        str2bool("boom")


@pytest.mark.parametrize(("provided", "expected"), [
    ("1", 1),
    ("None", None),
    ("none", None),
])
def test_optional_int(provided: str, expected: Optional[int]) -> None:
    assert optional_int(provided) == expected


@pytest.mark.parametrize(("provided", "expected"), [
    ("1.23", 1.23),
    ("1", 1),
    ("None", None),
    ("none", None),
])
def test_optional_float(provided: str, expected: Optional[float]) -> None:
    assert optional_float(provided) == expected
