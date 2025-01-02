import sys
import os
import pandas as pd
import numpy as np

new_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
if new_path not in sys.path:  # Avoid adding duplicates
    sys.path.append(new_path)
from utils.utils import find_best_params


def test_find_best_params_basic():
    # Create a simple DataFrame for testing
    data = {
        "param1": [1, 2, 3],
        "param2": [10, 20, 30],
        "target": [2, 8, 6],
    }
    metrics_df = pd.DataFrame(data)

    # Define parameters and expected result
    params = {"param1": 1, "param2": 10}
    target = "target"

    result = find_best_params(metrics_df, target, params, n=1)

    # Expected: The row with the highest target value (param1=2, param2=20)
    expected = {"param1": 3, "param2": 30}

    assert result == expected


def test_find_best_params_custom_agg():
    # Create a DataFrame
    data = {
        "param1": [1, 2, 3],
        "param2": [10, 20, 30],
        "target": [0.5, 0.8, 0.6],
    }
    metrics_df = pd.DataFrame(data)

    # Define parameters and custom aggregation function
    params = {"param1": 1, "param2": 10}
    target = "target"

    def agg_func(x):
        return np.median(x)

    result = find_best_params(
        metrics_df, target, params, n=1, agg_func=agg_func
    )

    # Expected: The row with the highest target value (param1=2, param2=20)
    expected = {"param1": 3, "param2": 30}

    assert result == expected


def test_find_best_params_no_neighbors():
    # Create a DataFrame with all targets equal
    data = {
        "param1": [1, 2, 3],
        "param2": [10, 20, 30],
        "target": [0.7, 0.5, 0.5],
    }
    metrics_df = pd.DataFrame(data)

    # Define parameters
    params = {"param1": 1, "param2": 10}
    target = "target"

    result = find_best_params(metrics_df, target, params, n=1)

    # Expected: The first row since all targets are equal (param1=1, param2=10)
    expected = {"param1": 1, "param2": 10}

    assert result == expected


def test_find_best_params_empty_dataframe():
    # Create an empty DataFrame
    metrics_df = pd.DataFrame(columns=["param1", "param2", "target"])

    # Define parameters
    params = {"param1": 1, "param2": 10}
    target = "target"

    result = find_best_params(metrics_df, target, params, n=1)

    # Expected: None (no data to evaluate)
    assert result is None
