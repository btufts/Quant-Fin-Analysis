import pytest
import pandas as pd
import sys
import os

new_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
if new_path not in sys.path:  # Avoid adding duplicates
    sys.path.append(new_path)
from utils import get_neighbors


@pytest.fixture
def sample_symbol_data():
    """Fixture to provide sample symbol data."""
    data = {
        "param1": [1, 2, 3, 4, 5],
        "param2": [10, 20, 30, 40, 50],
        "param3": [100, 200, 300, 400, 500],
    }
    return pd.DataFrame(data)


def test_get_neighbors_single_param(sample_symbol_data):
    row = pd.Series({"param1": 3, "param2": 30, "param3": 300})
    params = {"param1": 1}
    n = 1
    expected_neighbors = sample_symbol_data[
        (sample_symbol_data["param1"] >= 2)
        & (sample_symbol_data["param1"] <= 4)
    ]
    result = get_neighbors(sample_symbol_data, row, params, n)
    pd.testing.assert_frame_equal(result, expected_neighbors)


def test_get_neighbors_multiple_params(sample_symbol_data):
    row = pd.Series({"param1": 3, "param2": 30, "param3": 300})
    params = {"param1": 1, "param2": 10}
    n = 1
    expected_neighbors = sample_symbol_data[
        (sample_symbol_data["param1"] >= 2)
        & (sample_symbol_data["param1"] <= 4)
        & (sample_symbol_data["param2"] >= 20)
        & (sample_symbol_data["param2"] <= 40)
    ]
    result = get_neighbors(sample_symbol_data, row, params, n)
    pd.testing.assert_frame_equal(result, expected_neighbors)


def test_get_neighbors_no_matches(sample_symbol_data):
    row = pd.Series({"param1": 10, "param2": 100, "param3": 1000})
    params = {"param1": 1, "param2": 10}
    n = 1
    expected_neighbors = pd.DataFrame(
        columns=sample_symbol_data.columns, dtype=int
    )
    result = get_neighbors(sample_symbol_data, row, params, n)
    pd.testing.assert_frame_equal(result, expected_neighbors)


def test_get_neighbors_edge_case(sample_symbol_data):
    row = pd.Series({"param1": 1, "param2": 10, "param3": 100})
    params = {"param1": 1, "param2": 10, "param3": 100}
    n = 0
    expected_neighbors = sample_symbol_data[
        (sample_symbol_data["param1"] >= 1)
        & (sample_symbol_data["param1"] <= 1)
        & (sample_symbol_data["param2"] >= 10)
        & (sample_symbol_data["param2"] <= 10)
        & (sample_symbol_data["param3"] >= 100)
        & (sample_symbol_data["param3"] <= 100)
    ]
    result = get_neighbors(sample_symbol_data, row, params, n)
    pd.testing.assert_frame_equal(result, expected_neighbors)
