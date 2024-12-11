import pytest
import pandas as pd
import os
import sys

new_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
if new_path not in sys.path:  # Avoid adding duplicates
    sys.path.append(new_path)
from utils import reorder_csv


@pytest.fixture
def sample_csv_file(tmp_path):
    """Fixture to create a sample CSV file for testing."""
    file_path = tmp_path / "test.csv"
    data = {
        "Date": ["2024-01-01", "2024-01-02"],
        "Open": [100, 200],
        "High": [110, 210],
        "Low": [90, 190],
        "Close": [105, 205],
        "Adj Close": [104, 204],
        "Volume": [1000, 2000],
    }
    df = pd.DataFrame(data)
    df.to_csv(file_path, index=False)
    return file_path


@pytest.fixture
def sample_csv_with_datetime_file(tmp_path):
    """Fixture to create a sample CSV file with 'Datetime' column."""
    file_path = tmp_path / "test_datetime.csv"
    data = {
        "Datetime": ["2024-01-01 00:00:00", "2024-01-02 00:00:00"],
        "Open": [100, 200],
        "High": [110, 210],
        "Low": [90, 190],
        "Close": [105, 205],
        "Adj Close": [104, 204],
        "Volume": [1000, 2000],
    }
    df = pd.DataFrame(data)
    df.to_csv(file_path, index=False)
    return file_path


def test_reorder_csv(sample_csv_file):
    reorder_csv(sample_csv_file)
    df = pd.read_csv(sample_csv_file)

    # Verify the order of columns
    expected_columns = [
        "Date",
        "Open",
        "High",
        "Low",
        "Close",
        "Adj Close",
        "Volume",
    ]
    assert list(df.columns) == expected_columns


def test_reorder_csv_with_datetime(sample_csv_with_datetime_file):
    reorder_csv(sample_csv_with_datetime_file, datetime=True)
    df = pd.read_csv(sample_csv_with_datetime_file)

    # Verify the order of columns
    expected_columns = [
        "Datetime",
        "Open",
        "High",
        "Low",
        "Close",
        "Adj Close",
        "Volume",
    ]
    assert list(df.columns) == expected_columns


def test_reorder_csv_missing_columns(tmp_path):
    file_path = tmp_path / "test_missing.csv"
    data = {
        "Date": ["2024-01-01", "2024-01-02"],
        "Open": [100, 200],
        "High": [110, 210],
        "Close": [105, 205],
    }  # Missing 'Low', 'Adj Close', 'Volume'
    df = pd.DataFrame(data)
    df.to_csv(file_path, index=False)

    with pytest.raises(ValueError, match="Missing columns:"):
        reorder_csv(file_path)


def test_reorder_csv_empty_file(tmp_path):
    file_path = tmp_path / "empty.csv"
    pd.DataFrame().to_csv(file_path, index=False)

    with pytest.raises(ValueError, match="No columns to parse from file"):
        reorder_csv(file_path)
