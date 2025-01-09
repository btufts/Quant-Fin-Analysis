"""
Download data from yahoo finance and save it to a csv file
"""

# System imports
from datetime import datetime, timedelta
import argparse

# Third party imports
import yfinance as yf
import pandas as pd


def get_interval_timedelta(interval: str) -> timedelta:
    """
    Return an appropriate `timedelta` object corresponding to the given interval.

    Examples of supported intervals in yfinance:
        - 1m, 2m, 5m, 15m, 30m, 60m (1h), 90m, 1h
        - 1d, 5d
        - 1wk
        - 1mo, 3mo
    """
    # We'll define some basic mappings. Feel free to extend as needed.
    # Note that yfinance uses '60m' as an alias for '1h', so you may need to handle that equivalently.
    mapping = {
        "1m": timedelta(minutes=1),
        "2m": timedelta(minutes=2),
        "5m": timedelta(minutes=5),
        "15m": timedelta(minutes=15),
        "30m": timedelta(minutes=30),
        "60m": timedelta(hours=1),
        "90m": timedelta(minutes=90),
        "1h": timedelta(hours=1),  # same as "60m"
        "1d": timedelta(days=1),
        "5d": timedelta(days=5),
        "1wk": timedelta(weeks=1),
        "1mo": timedelta(days=30),  # approximate
        "3mo": timedelta(days=90),  # approximate
    }
    if interval not in mapping:
        # Fallback: treat as 1 day
        return timedelta(days=1)
    return mapping[interval]


def main(
    ticker: str,
    start_date: str,
    end_date: str,
    interval: str,
    ignore_tz: bool,
    existing_file="",
    output_path="",
):
    """
    Update a CSV file with new data from Yahoo Finance.

    Modifications made:
      1. For intraday intervals (e.g., 1h, 1m, 1s, ...), ensure start_date is no older than 730 days.
      2. When determining the new start date from existing CSV data, increment it by the requested interval (instead of a fixed 1-day increment).
    """

    input_start_date = datetime.strptime(start_date, "%Y-%m-%d")
    input_end_date = datetime.strptime(end_date, "%Y-%m-%d")

    def is_intraday(interval_str: str) -> bool:
        # yfinance intraday intervals typically contain 'm' or 'h': 1m, 5m, 30m, 1h, 60m, etc.
        return any(x in interval_str for x in ["m", "h", "s"])

    if is_intraday(interval):
        earliest_start_allowed = datetime.now() - timedelta(days=729)
        if input_start_date < earliest_start_allowed:
            print(
                f"Requested start date {input_start_date.date()} is too old for intraday data. "
                f"Using {earliest_start_allowed.date()} instead."
            )
            input_start_date = earliest_start_allowed

    if existing_file:
        try:
            existing_data = pd.read_csv(
                existing_file, parse_dates=True, index_col="Datetime"
            )
            print("Existing data loaded.")
        except FileNotFoundError:
            raise FileNotFoundError(f"File {existing_file} not found.")
    else:
        existing_data = pd.DataFrame()

    if not existing_data.empty:
        latest_date_in_csv = existing_data.index[-1]
        # Use a timedelta that matches the requested interval
        interval_td = get_interval_timedelta(interval)
        start_date_dt = latest_date_in_csv + interval_td
        start_date_dt = max(start_date_dt, input_start_date)
    else:
        # If no existing data, start from the user-provided date
        start_date_dt = input_start_date

    # Also ensure the final end_date_dt is not before the new start_date_dt
    end_date_dt = max(start_date_dt, input_end_date)

    start_date_str = start_date_dt.strftime("%Y-%m-%d")
    end_date_str = end_date_dt.strftime("%Y-%m-%d")

    print(
        f"Fetching new data from {start_date_str} to {end_date_str} (interval={interval})."
    )
    new_data = yf.download(
        ticker,
        start=start_date_str,
        end=end_date_str,
        interval=interval,
        ignore_tz=ignore_tz,
    )

    if new_data.empty:
        print("No new data found.")
        # If no new data, just return existing data
        return existing_data

    # 7. Prepare new data and merge
    new_data.index.name = "Datetime"

    # Convert datetime index to remove timezone
    new_data.index = new_data.index.tz_localize(None)

    # Concatenate old and new data and drop duplicates
    updated_data = pd.concat([existing_data, new_data])

    updated_data = updated_data[~updated_data.index.duplicated(keep="last")]

    # 8. Save the updated data
    if not output_path:
        updated_data.to_csv(existing_file)
        print(f"Data updated and saved to {existing_file}.")
    else:
        updated_data.to_csv(output_path)
        print(f"Data updated and saved to {output_path}.")

    return updated_data


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download data from yahoo finance and save it to a csv file."
    )

    parser.add_argument(
        "--ticker",
        type=str,
        required=True,
        help="The ticker symbol to fetch data for.",
    )

    parser.add_argument(
        "--start_date",
        type=str,
        default="2000-01-01",
        help="The start date for fetching data. Default is 2000-01-01.",
    )

    parser.add_argument(
        "--end_date",
        type=str,
        default=datetime.now().strftime("%Y-%m-%d"),
        help="The end date for fetching data. Default is today's date.",
    )

    parser.add_argument(
        "--interval",
        type=str,
        default="1d",
        help="The interval for fetching data. Default is 1d.",
    )

    # TODO: Add support for keeping timezone when fetching data
    # Backtrader does not support timezone so this data will not be
    # compatible with backtrader
    parser.add_argument(
        "--ignore_tz",
        action="store_true",
        help="Ignore timezone when fetching data. Default is False.",
    )

    parser.add_argument(
        "--existing_file",
        type=str,
        help="Path to an existing CSV file with historical data to extend.",
    )

    parser.add_argument(
        "--output_path",
        type=str,
        help="Path to save the updated data to a new CSV file.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if not args.existing_file and not args.output_path:
        raise ValueError(
            "Either --existing_file or --output_path must be provided."
        )

    main(**vars(args))
