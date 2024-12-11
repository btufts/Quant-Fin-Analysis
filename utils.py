# System Imports
from __future__ import (
    absolute_import,
    division,
    print_function,
    unicode_literals,
)
from typing import Callable

# Third party imports
import pandas as pd

import collections
from datetime import datetime
import io
import itertools

from backtrader import feed
from backtrader.utils import date2num


# This is a copy from backtrader with some
# modifications to be able to have datetime
# instead of just date
class YahooFinanceCSVData(feed.CSVDataBase):
    lines = ("adjclose",)

    params = (
        ("reverse", False),
        ("adjclose", True),
        ("adjvolume", True),
        ("round", True),
        ("decimals", 2),
        ("roundvolume", False),
        ("swapcloses", False),
    )

    def start(self):
        super(YahooFinanceCSVData, self).start()

        if not self.params.reverse:
            return

        dq = collections.deque()
        for line in self.f:
            dq.appendleft(line)

        f = io.StringIO(newline=None)
        f.writelines(dq)
        f.seek(0)
        self.f.close()
        self.f = f

    def _loadline(self, linetokens):
        while True:
            nullseen = False
            for tok in linetokens[1:]:
                if tok == "null":
                    nullseen = True
                    linetokens = self._getnextline()
                    if not linetokens:
                        return False

                    break

            if not nullseen:
                break

        i = itertools.count(0)

        dttxt = linetokens[next(i)]
        dt = datetime.strptime(dttxt, "%Y-%m-%d %H:%M:%S")
        dtnum = date2num(dt)

        self.lines.datetime[0] = dtnum
        o = float(linetokens[next(i)])
        h = float(linetokens[next(i)])
        low = float(linetokens[next(i)])
        c = float(linetokens[next(i)])
        self.lines.openinterest[0] = 0.0

        adjustedclose = float(linetokens[next(i)])
        try:
            v = float(linetokens[next(i)])
        except ValueError:
            v = 0.0

        if self.p.swapcloses:
            c, adjustedclose = adjustedclose, c

        adjfactor = c / adjustedclose

        if self.params.adjclose:
            o /= adjfactor
            h /= adjfactor
            low /= adjfactor
            c = adjustedclose
            if self.p.adjvolume:
                v *= adjfactor

        if self.p.round:
            decimals = self.p.decimals
            o = round(o, decimals)
            h = round(h, decimals)
            low = round(low, decimals)
            c = round(c, decimals)

        v = round(v, self.p.roundvolume)

        self.lines.open[0] = o
        self.lines.high[0] = h
        self.lines.low[0] = low
        self.lines.close[0] = c
        self.lines.volume[0] = v
        self.lines.adjclose[0] = adjustedclose

        return True


def get_neighbors(symbol_data, row, params, n):
    """
    Get the neighbors of a given row within a range of parameters.

    Parameters
    ----------
    symbol_data : pd.DataFrame
        The dataframe containing the data for the symbol.
    row : pd.Series
        The row for which to find the neighbors.
    params : dict[str, int]
        A dictionary mapping parameter names to the steps to take
        in each parameter to consider a neighbor.
    n : int, default=1
        The number of steps to take in each parameter to consider a neighbor.

    Returns
    -------
    neighbors : pd.DataFrame
        A dataframe containing the neighbors of the given row.
    """
    # Initialize the condition to True so we can
    # apply multiple conditions using & operator
    condition = True
    for param, step in params.items():
        condition &= symbol_data[param].between(
            row[param] - (n * step), row[param] + (n * step)
        )
    # Filter based on the dynamic condition
    neighbors = symbol_data[condition]
    return neighbors


def find_best_params(
    metrics_df: pd.DataFrame,
    target: str,
    params: dict[str, int],
    n: int = 1,
    agg_func: Callable[[pd.Series], float] = pd.Series.mean,
) -> dict[str, tuple]:
    """
    Find the best parameters in the given metrics_df based on the target
    column and the given parameters with their respective steps.

    Parameters
    ----------
    metrics_df: pd.DataFrame
        The DataFrame containing the metrics to be analyzed. It should have
        columns for the parameters and a column for the target.

    target: str
        The name of the column to be used as the target.

    params: dict[str, int]
        A dictionary where the keys are the names of the columns that should
        be used as parameters and the values are the steps
        to be used when finding the neighbors.

    n: int, optional
        The number of neighbors to consider when finding the best parameters.
        Defaults to 1.

    agg_func: Callable[[pd.Series], float], optional
        The aggregation function to be used when calculating the average
        target. Defaults to pd.Series.mean.

    Returns
    -------
    dict[str, tuple]
        A dictionary where the keys are the names of the parameters and the
        values are tuples containing the best value and the standard
        deviation of the best values found.
    """
    best_avg_target = float("-inf")
    curr_best = None

    # Iterate through each row for the current symbol
    for _, row in metrics_df.iterrows():
        neighbors = get_neighbors(metrics_df, row, params, n)

        # Calculate the average target of the current point and its neighbors
        avg_target = agg_func(neighbors[target])

        # Update the best triple if the current average target is higher
        if avg_target > best_avg_target:
            best_avg_target = avg_target
            curr_best = {param: row[param] for param in params.keys()}

    return curr_best


def reorder_csv(file_path, datetime=False):
    # Read the CSV file
    """
    Reorders a CSV file with the expected column order.

    Parameters
    ----------
    file_path : str
        The path to the CSV file to be reordered.
    datetime : bool, optional
        If True, assumes the date column is named "Datetime" instead of "Date".
        Defaults to False.

    Raises
    ------
    ValueError
        If any of the expected columns are missing from the CSV file.

    Notes
    -----
    The expected column order is:
    ["Date"/"Datetime", "Open", "High", "Low", "Close", "Adj Close", "Volume"]

    """
    df = pd.read_csv(file_path)

    date_title = "Date"
    if datetime:
        date_title = "Datetime"

    # Expected column order
    expected_order = [
        date_title,
        "Open",
        "High",
        "Low",
        "Close",
        "Adj Close",
        "Volume",
    ]

    # Check if all expected columns are present
    missing_columns = set(expected_order) - set(df.columns)
    if missing_columns:
        raise ValueError(f"Missing columns: {missing_columns}")

    # Reorder the columns
    df = df[expected_order]

    # Save the reordered DataFrame back to the same CSV file
    df.to_csv(file_path, index=False)
    print(f"File {file_path} has been successfully reordered and saved.")


def print_strategy_stats(strategy):
    # Extract statistics from analyzers
    """
    Prints a markdown table containing various statistics about the given
    strategy.

    The statistics include:
    - Initial and ending capital
    - Net profit and net profit percentage
    - Exposure percentage
    - Net risk adjusted return
    - Annual return percentage
    - Risk adjusted return percentage
    - Transaction costs
    - Various trade statistics (total number of trades, average profit/loss,
    etc.)
    - Various metrics about the strategy's performance (max trade drawdown,
    max system drawdown, etc.)

    Parameters
    ----------
    strategy : bt.Strategy
        The strategy to generate statistics for.

    Returns
    -------
    str
        A markdown table containing the statistics.
    """
    returns_stats = strategy.analyzers.returns.get_analysis()
    trade_stats = strategy.analyzers.trade_stats.get_analysis()
    drawdown_stats = strategy.analyzers.drawdown.get_analysis()
    in_market_stats = strategy.analyzers.in_market.get_analysis()

    # Calculate derived metrics
    total_in_market_bars = in_market_stats.get("Total In-Market Bars", 0)
    total_bars = in_market_stats.get("Total Bars", 1)
    total_gains = in_market_stats.get("Total Gains", 1)
    total_losses = in_market_stats.get("Total Losses", 1)
    time_in_market = total_in_market_bars / total_bars if total_bars > 0 else 0
    return_per_exposure = (
        returns_stats["rnorm"] / time_in_market if time_in_market > 0 else 0
    )
    profit_factor = total_gains / total_losses if total_losses > 0 else "N/A"
    avg_gain = (
        in_market_stats.get("Total Percent Gain", 0) / trade_stats.total.closed
        if trade_stats.total.closed > 0
        else "N/A"
    )

    # Assemble data into a markdown table
    markdown_table = f"""
| Metric                          | Value         |
|---------------------------------|---------------|
| Initial Capital                 | {strategy.broker.startingcash:.2f}     |
| Ending Capital                  | {strategy.broker.getvalue():.2f}      |
| Net Profit                      | {strategy.broker.getvalue() -
                                     strategy.broker.startingcash:.2f} |
| Net Profit %                    | {returns_stats['rtot'] * 100:.2f}%    |
| Exposure %                      | {time_in_market * 100:.2f}%           |
| Net Risk Adjusted Return        | {return_per_exposure * 100:.2f}%      |
| Annual Return %                 | {returns_stats['rnorm'] * 100:.2f}%   |
| Risk Adjusted Return %          | {return_per_exposure * 100:.2f}%      |
| Transaction Costs               | 0.00                                  |

| All Trades                      |               |
|---------------------------------|---------------|
| Total Number of Trades          | {trade_stats.total.closed}            |
| Avg. Profit/Loss                | {avg_gain:.2f}                        |
| Avg. Profit/Loss %              | {((returns_stats['rtot'] * 100) /
                                      trade_stats.total.closed):.2f}%|
| Avg. Bars Held                  | N/A                                   |

| Winners                         |               |
|---------------------------------|---------------|
| Total Profit                    | {trade_stats.won.pnl.total:.2f}         |
| Avg. Profit                     | {trade_stats.won.pnl.average:.2f}     |
| Avg. Profit %                   | N/A                                   |
| Avg. Bars Held                  | N/A                                   |
| Max. Consecutive Wins           | {trade_stats.streak.won.longest}      |
| Largest Win                     | N/A                                   |
| Bars in Largest Win             | N/A                                   |

| Losers                          |               |
|---------------------------------|---------------|
| Total Loss                      | {trade_stats.lost.pnl.total:.2f}        |
| Avg. Loss                       | {trade_stats.lost.pnl.average:.2f}    |
| Avg. Loss %                     | N/A                                   |
| Avg. Bars Held                  | N/A                                   |
| Max. Consecutive Losses         | {trade_stats.streak.lost.longest}     |
| Largest Loss                    | N/A                                   |
| Bars in Largest Loss            | N/A                                   |

| Additional Metrics              |               |
|---------------------------------|---------------|
| Max. Trade Drawdown             | N/A                                   |
| Max. Trade % Drawdown           |                                       |
| Max. System Drawdown            | {drawdown_stats.max.moneydown:.2f}     |
| Max. System % Drawdown          | {drawdown_stats.max.drawdown:.2f}% |
| Recovery Factor                 | N/A                                   |
| CAR/MaxDD                       | N/A                                   |
| RAR/MaxDD                       | N/A                                   |
| Profit Factor                   | {profit_factor}                       |
| Payoff Ratio                    | N/A                                   |
| Standard Error                  | N/A                                   |
| Risk-Reward Ratio               | N/A                                   |
"""
    return markdown_table


def get_strategy_stats(strategy, flatten=False, opt=False):
    # Extract statistics from analyzers
    """
    Extracts and calculates statistics from a Backtrader strategy.

    Args:
        strategy (backtrader.Strategy): The strategy to extract statistics
        from.
        flatten (bool, optional): If True, flattens the returned dictionary.
        Defaults to False.
        opt (bool, optional): If True, omits certain fields from the returned
        dictionary, such as the initial and ending capital. Defaults to False.

    Returns:
        dict: A dictionary containing the extracted statistics.
    """
    returns_stats = strategy.analyzers.returns.get_analysis()
    trade_stats = strategy.analyzers.trade_stats.get_analysis()
    drawdown_stats = strategy.analyzers.drawdown.get_analysis()
    in_market_stats = strategy.analyzers.in_market.get_analysis()

    if trade_stats.total.total == 0:
        return None

    # Calculate derived metrics
    total_in_market_bars = in_market_stats.get("Total In-Market Bars", 0)
    total_bars = in_market_stats.get("Total Bars", 1)
    total_gains = in_market_stats.get("Total Gains", 1)
    total_losses = in_market_stats.get("Total Losses", 1)
    time_in_market = total_in_market_bars / total_bars if total_bars > 0 else 0
    return_per_exposure = (
        returns_stats["rnorm"] / time_in_market if time_in_market > 0 else 0
    )
    profit_factor = total_gains / total_losses if total_losses > 0 else None
    avg_gain = (
        in_market_stats.get("Total Percent Gain", 0) / trade_stats.total.closed
        if trade_stats.total.closed > 0
        else None
    )

    # Assemble data into a structured dictionary
    stats_dict = {
        "capital": {
            "initial": strategy.broker.startingcash if not opt else None,
            "ending": strategy.broker.getvalue() if not opt else None,
            "net_profit": (
                strategy.broker.getvalue() - strategy.broker.startingcash
                if not opt
                else None
            ),
            "net_profit_percent": returns_stats["rtot"] * 100,
        },
        "exposure": {
            "percent": time_in_market * 100,
            "net_risk_adjusted_return": (
                return_per_exposure * 100 if return_per_exposure else 0.0
            ),
        },
        "returns": {
            "cagr": returns_stats["rnorm"] * 100,
            "sharpe_ratio": (
                strategy.analyzers.sharpe.get_analysis()["sharperatio"]
                if strategy.analyzers.sharpe.get_analysis()["sharperatio"]
                else 0.0
            ),
        },
        "trades": {
            "total": trade_stats.total.closed,
            "avg_profit_loss": (avg_gain if avg_gain is not None else 0.0),
            "avg_profit_loss_percent": (
                (returns_stats["rtot"] * 100) / trade_stats.total.closed
                if trade_stats.total.closed > 0
                else 0.0
            ),
            "winners": {
                "total_profit": trade_stats.won.pnl.total,
                "avg_profit": trade_stats.won.pnl.average,
                "max_consecutive": trade_stats.streak.won.longest,
            },
            "losers": {
                "total_loss": trade_stats.lost.pnl.total,
                "avg_loss": trade_stats.lost.pnl.average,
                "max_consecutive": trade_stats.streak.lost.longest,
            },
        },
        "drawdown": {
            "max_money": drawdown_stats.max.moneydown,
            "max_percent": drawdown_stats.max.drawdown,
            "max_drawdown_duration": drawdown_stats.max.len,
        },
        "profit_factor": (profit_factor if profit_factor is not None else 0.0),
    }

    if flatten:
        stats_dict = flatten_dict(stats_dict)
        stats_dict = {k: v for k, v in stats_dict.items() if v is not None}

    return stats_dict


def flatten_dict(d, parent_key="", sep="."):
    """
    Flattens a nested dictionary into a one-level dictionary.

    Args:
        d (dict): The dictionary to be flattened.
        parent_key (str, optional): The parent key to be prepended to the new
        keys. Defaults to "".
        sep (str, optional): The separator to be used when joining the parent
        key and the current key. Defaults to ".".

    Returns:
        dict: The flattened dictionary.
    """
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):  # If the value is a nested dictionary
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:  # If the value is not a dictionary
            items.append((new_key, v))
    return dict(items)


def get_buy_and_hold(filename, initial_investment=100000, datetime=False):
    """
    Reads a CSV file containing a stock's historical data and calculates the
    buy and hold value.

    Args:
        filename (str): The path to the CSV file.
        initial_investment (int, optional): The initial amount of money to
        invest. Defaults to 100000.
        datetime (bool, optional): If True, the 'Date' column is treated as a
        datetime object and the 'Datetime' column is created. Defaults to
        False.

    Returns:
        pandas.DataFrame: A DataFrame containing the original data and a new
        'BuyAndHoldValue' column.
    """
    data = pd.read_csv(filename)

    if datetime:
        data["Datetime"] = pd.to_datetime(data["Datetime"])
        data["Date"] = data["Datetime"].dt.date
    else:
        data["Date"] = pd.to_datetime(data["Date"])

    # Check if 'Adj Close' column exists
    if "Adj Close" not in data.columns:
        raise ValueError("The file must contain an 'Adj Close' column.")

    # Calculate the buy and hold value
    initial_price = data["Adj Close"].iloc[0]
    data["BuyAndHoldValue"] = (
        data["Adj Close"] / initial_price
    ) * initial_investment

    return data
