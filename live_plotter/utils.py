from datetime import datetime
from typing import List, Union


def assert_equals(a, b):
    assert a == b, f"{a} != {b}"


def datetime_str() -> str:
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def convert_to_list_str_fixed_len(
    str_or_list_str: Union[str, List[str]], fixed_length: int
) -> List[str]:
    if isinstance(str_or_list_str, str):
        return [str_or_list_str] * fixed_length

    if len(str_or_list_str) < fixed_length:
        return str_or_list_str + [""] * (fixed_length - len(str_or_list_str))

    return str_or_list_str[:fixed_length]
