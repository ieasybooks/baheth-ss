import unicodedata

from typing import Any


def remove_control_characters(value: Any) -> Any:
    if type(value) == str:
        return ''.join(ch for ch in value if unicodedata.category(ch)[0] != 'C')

    return value


def strip_str(value: Any) -> Any:
    if type(value) == str:
        return ' '.join(value.strip().split())

    return value


def to_int_if_float_or_str(value: Any) -> Any:
    try:
        return str(int(float(value)))
    except ValueError:
        return value
