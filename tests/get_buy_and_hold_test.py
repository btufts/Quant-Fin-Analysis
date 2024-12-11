import pytest
import pandas as pd
import os
import sys
from math import isclose

new_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
if new_path not in sys.path:  # Avoid adding duplicates
    sys.path.append(new_path)
from utils import get_buy_and_hold


@pytest.fixture
def sample_csv_file(tmp_path):
    """Fixture to create a sample CSV file for testing."""
    file_path = tmp_path / "test.csv"
    data = {
        "Date": ["2024-01-01", "2024-01-02", "2024-01-03"],
        "Adj Close": [100, 110, 120],
    }
    df = pd.DataFrame(data)
    df.to_csv(file_path, index=False)
    return file_path


@pytest.fixture
def sample_csv_with_datetime_file(tmp_path):
    """Fixture to create a sample CSV file with 'Datetime' column."""
    file_path = tmp_path / "test_datetime.csv"
    data = {
        "Datetime": [
            "2024-01-01 00:00:00",
            "2024-01-02 00:00:00",
            "2024-01-03 00:00:00",
        ],
        "Adj Close": [100, 110, 120],
    }
    df = pd.DataFrame(data)
    df.to_csv(file_path, index=False)
    return file_path


def test_get_buy_and_hold(sample_csv_file):
    df = get_buy_and_hold(sample_csv_file)

    # Verify the 'BuyAndHoldValue' column is calculated correctly
    expected_values = [100000.0, 110000.0, 120000.0]
    assert all(
        isclose(a, b, rel_tol=0.0001, abs_tol=0.0001)
        for a, b in zip(list(df["BuyAndHoldValue"]), expected_values)
    ), "Lists are not approximately equal"


def test_get_buy_and_hold_with_datetime(sample_csv_with_datetime_file):
    df = get_buy_and_hold(sample_csv_with_datetime_file, datetime=True)

    # Verify the 'BuyAndHoldValue' column is calculated correctly
    expected_values = [100000, 110000, 120000]
    assert all(
        isclose(a, b, rel_tol=0.0001, abs_tol=0.0001)
        for a, b in zip(list(df["BuyAndHoldValue"]), expected_values)
    ), "Lists are not approximately equal"


def test_get_buy_and_hold_missing_adj_close(tmp_path):
    file_path = tmp_path / "test_missing_adj_close.csv"
    data = {
        "Date": ["2024-01-01", "2024-01-02", "2024-01-03"],
        "Close": [100, 110, 120],  # Missing 'Adj Close'
    }
    df = pd.DataFrame(data)
    df.to_csv(file_path, index=False)

    with pytest.raises(
        ValueError, match="The file must contain an 'Adj Close' column."
    ):
        get_buy_and_hold(file_path)


def test_get_buy_and_hold_empty_file(tmp_path):
    file_path = tmp_path / "empty.csv"
    pd.DataFrame().to_csv(file_path, index=False)

    with pytest.raises(pd.errors.EmptyDataError):
        get_buy_and_hold(file_path)


def test_get_buy_and_hold_custom_investment(sample_csv_file):
    df = get_buy_and_hold(sample_csv_file, initial_investment=50000)

    # Verify the 'BuyAndHoldValue' column is calculated correctly with a custom investment
    expected_values = [50000, 55000, 60000]
    assert all(
        isclose(a, b, rel_tol=0.0001, abs_tol=0.0001)
        for a, b in zip(list(df["BuyAndHoldValue"]), expected_values)
    ), "Lists are not approximately equal"
