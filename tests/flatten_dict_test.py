import sys
import os

new_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
if new_path not in sys.path:  # Avoid adding duplicates
    sys.path.append(new_path)
from utils.utils import flatten_dict


def test_flatten_dict_simple():
    input_dict = {"a": 1, "b": 2}
    expected_output = {"a": 1, "b": 2}
    assert flatten_dict(input_dict) == expected_output


def test_flatten_dict_nested():
    input_dict = {"a": {"b": {"c": 1}, "d": 2}}
    expected_output = {"a.b.c": 1, "a.d": 2}
    assert flatten_dict(input_dict) == expected_output


def test_flatten_dict_with_separator():
    input_dict = {"a": {"b": {"c": 1}, "d": 2}}
    expected_output = {"a-b-c": 1, "a-d": 2}
    assert flatten_dict(input_dict, sep="-") == expected_output


def test_flatten_dict_with_parent_key():
    input_dict = {"b": {"c": 1}, "d": 2}
    expected_output = {"x.b.c": 1, "x.d": 2}
    assert flatten_dict(input_dict, parent_key="x") == expected_output


def test_flatten_dict_empty():
    input_dict = {}
    expected_output = {}
    assert flatten_dict(input_dict) == expected_output


def test_flatten_dict_mixed_types():
    input_dict = {
        "a": {"b": 1, "c": [1, 2, 3], "d": {"e": "hello"}},
        "f": True,
    }
    expected_output = {"a.b": 1, "a.c": [1, 2, 3], "a.d.e": "hello", "f": True}
    assert flatten_dict(input_dict) == expected_output
