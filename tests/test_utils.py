import pytest

from whisper.utils import str2bool


@pytest.mark.parametrize(
    "value, expected",
    [
        ("True", True),
        ("False", False),
        ("true", True),
        ("false", False),
        ("TRUE", True),
        ("FALSE", False),
        ("1", True),
        ("0", False),
        ("yes", True),
        ("no", False),
    ],
)
def test_str2bool_accepts_common_boolean_spellings(value, expected):
    assert str2bool(value) is expected


def test_str2bool_rejects_invalid_values():
    with pytest.raises(ValueError, match="Expected a boolean value"):
        str2bool("maybe")
