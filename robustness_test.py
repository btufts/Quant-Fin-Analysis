# System Imports
import argparse
import os
import random
from collections import defaultdict

# Third Party Imports
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd

# Local Imports
from optimize_strategy import backtest
from strategies import (
    RandomStrategy,
    WilliamsRStrategy,
    CCIStrategy,
    StochasticStrategy,
    TurnaroundTuesday,
)
import utils.utils as utils


def strategy_class(strategy):
    if strategy == "WilliamsR":
        return WilliamsRStrategy
    elif strategy == "Random":
        return RandomStrategy
    elif strategy == "CCI":
        return CCIStrategy
    elif strategy == "Stochastic":
        return StochasticStrategy
    elif strategy == "Turnaround":
        return TurnaroundTuesday
    else:
        raise ValueError(f"Unknown strategy: {strategy}")


def vsrandom(args):
    """
    Compare the strategy against a random strategy. This function takes the
    same arguments as backtest, but instead of running a single backtest, it
    runs the strategy against a random strategy multiple times and plots a
    comparison curve.

    The function also saves a CSV of the results.

    Parameters
    ----------
    args : argparse.Namespace
        The arguments passed to the backtest function.

    Returns
    -------
    None
    """
    params = (
        pd.read_csv(args.strategy_report + "/best_parameters.csv")
        .set_index("symbol")
        .to_dict(orient="index")
    )

    data_results = defaultdict(list)
    symbol = args.data.split(".")[0]
    symbol = args.symbol
    random_results = []
    for _ in tqdm(range(args.vsrandom_itrs), desc="Running vsrandom test"):
        random_results.append(
            backtest(f"{args.data}", RandomStrategy, {}, args)
        )

    plt.figure(figsize=(10, 6), dpi=100)
    plt.axhline(y=100000, color="black", linestyle="--", linewidth=2)

    for i, strat in enumerate(random_results):
        flattened_results = utils.get_strategy_stats(
            strat[0], flatten=True, opt=True
        )
        if flattened_results is None:
            raise RuntimeError("The strategy made no trades in the test data")
        data_results["symbol"].append(symbol)
        for key, value in flattened_results.items():
            data_results[key].append(value)
        for param in params[symbol]:
            data_results[param].append(-1)

        plt.plot(
            strat[0].datetimes,
            strat[0].equity_curve,
            color="lightgrey",
            alpha=0.5,
            zorder=1,
        )

    strategy_results = backtest(
        f"{args.data}", strategy_class(args.strategy), params[symbol], args
    )
    flattened_results = utils.get_strategy_stats(
        strategy_results[0], flatten=True, opt=True
    )
    if flattened_results is None:
        raise RuntimeError("The strategy made no trades in the test data")
    data_results["symbol"].append(symbol)
    for key, value in flattened_results.items():
        data_results[key].append(value)
    for param in params[symbol]:
        data_results[param].append(getattr(strategy_results[0].params, param))

    plt.plot(
        strategy_results[0].datetimes,
        strategy_results[0].equity_curve,
        color="red",
        alpha=1,
        zorder=3,
    )

    # Set the y-ticks and labels
    plt.title(f"{args.strategy} Equity Curve vs Random")
    plt.xlabel("Hours")
    plt.ylabel("Portfolio Value ($USD)")

    plt.yscale("log")
    # plt.legend(fontsize='x-small')
    plt.tight_layout()
    if not os.path.exists(args.strategy_report + "/vsrandom/"):
        os.makedirs(args.strategy_report + "/vsrandom/")
    plt.savefig(f"{args.strategy_report}/vsrandom/{symbol}_vsrandom_curve.png")
    pd.DataFrame(data_results).to_csv(
        f"{args.strategy_report}/vsrandom/{symbol}_results.csv",
        index=False,
    )


def mc_randomized_entry(args):
    """
    Perform a Monte Carlo simulation by applying a randomized entry condition
    to a trading strategy and comparing it against the original strategy.

    The function modifies the long entry condition of the given strategy to
    use a random decision process. It then runs several iterations of the
    strategy with this randomized entry to assess its robustness.

    Parameters
    ----------
    args : argparse.Namespace
        The arguments containing the configuration for the backtest, including
        the strategy, data, and number of iterations.

    Returns
    -------
    None
    The results, including equity curves and strategy statistics, are saved as
    plots and CSV files in the specified directory.
    """
    params = (
        pd.read_csv(args.strategy_report + "/best_parameters.csv")
        .set_index("symbol")
        .to_dict(orient="index")
    )
    strategy = strategy_class(args.strategy)

    def randomized_entry(self, *args, **kwargs):
        action = random.randint(0, 2)
        if action:
            return True
        else:
            return False

    # Get original results
    strategy_results = None
    random_results = []
    symbol = args.symbol
    strategy_results = backtest(f"{args.data}", strategy, params[symbol], args)

    strategy.long_condition = randomized_entry
    for _ in tqdm(range(args.mcrandom_itrs), desc="Running mcrandom test"):
        random_results.append(
            backtest(f"{args.data}", strategy, params[symbol], args)
        )

    data_results = defaultdict(list)
    plt.figure(figsize=(10, 6), dpi=100)
    plt.axhline(y=100000, color="black", linestyle="--", linewidth=2)

    plt.plot(
        strategy_results[0].datetimes,
        strategy_results[0].equity_curve,
        color="red",
        alpha=1,
        zorder=3,
    )
    flattened_results = utils.get_strategy_stats(
        strategy_results[0], flatten=True, opt=True
    )
    if flattened_results is None:
        raise RuntimeError("The strategy made no trades in the test data")
    data_results["symbol"].append(symbol)
    for key, value in flattened_results.items():
        data_results[key].append(value)
    for param in params[symbol]:
        data_results[param].append(getattr(strategy_results[0].params, param))

    for i, strat in enumerate(random_results):
        flattened_results = utils.get_strategy_stats(
            strat[0], flatten=True, opt=True
        )
        if flattened_results is None:
            raise RuntimeError("The strategy made no trades in the test data")
        data_results["symbol"].append(symbol)
        for key, value in flattened_results.items():
            data_results[key].append(value)
        for param in params[symbol]:
            data_results[param].append(getattr(strat[0].params, param))

        plt.plot(
            strat[0].datetimes,
            strat[0].equity_curve,
            color="lightgrey",
            alpha=0.5,
            zorder=1,
        )
    # Set the y-ticks and labels
    plt.title(f"{args.strategy} Equity Curve vs Random Entry")
    plt.xlabel("Hours")
    plt.ylabel("Portfolio Value ($USD)")

    plt.yscale("log")
    # plt.legend(fontsize='x-small')
    plt.tight_layout()
    if not os.path.exists(args.strategy_report + "/mcrandomentry/"):
        os.makedirs(args.strategy_report + "/mcrandomentry/")
    plt.savefig(
        f"{args.strategy_report}/mcrandomentry"
        + f"/{symbol}_mcrandomentry_curve.png"
    )
    pd.DataFrame(data_results).to_csv(
        f"{args.strategy_report}/mcrandomentry/{symbol}_results.csv",
        index=False,
    )


def mc_randomized_exit(args):
    """
    Perform a Monte Carlo simulation by applying a randomized exit condition
    to a trading strategy and comparing it against the original strategy.

    The function modifies the exit condition of the given strategy to use a
    random decision process. It then runs several iterations of the strategy
    with this randomized exit to assess its robustness.

    Parameters
    ----------
    args : argparse.Namespace
        The arguments containing the configuration for the backtest, including
        the strategy, data, and number of iterations.

    Returns
    -------
    None
    The results, including equity curves and strategy statistics, are saved as
    plots and CSV files in the specified directory.
    """
    params = (
        pd.read_csv(args.strategy_report + "/best_parameters.csv")
        .set_index("symbol")
        .to_dict(orient="index")
    )
    strategy = strategy_class(args.strategy)

    def randomized_exit(self, *args, **kwargs):
        action = random.randint(0, 2)
        if action:
            return True
        else:
            return False

    # Get original results
    # Get original results
    strategy_results = None
    random_results = []
    symbol = args.symbol
    strategy_results = backtest(f"{args.data}", strategy, params[symbol], args)

    strategy.long_condition = randomized_exit
    for _ in tqdm(range(args.mcrandom_itrs), desc="Running mcrandom test"):
        random_results.append(
            backtest(f"{args.data}", strategy, params[symbol], args)
        )

    data_results = defaultdict(list)
    plt.figure(figsize=(10, 6), dpi=100)
    plt.axhline(y=100000, color="black", linestyle="--", linewidth=2)

    plt.plot(
        strategy_results[0].datetimes,
        strategy_results[0].equity_curve,
        color="red",
        alpha=1,
        zorder=3,
    )

    flattened_results = utils.get_strategy_stats(
        strategy_results[0], flatten=True, opt=True
    )
    if flattened_results is None:
        raise RuntimeError("The strategy made no trades in the test data")
    data_results["symbol"].append(symbol)
    for key, value in flattened_results.items():
        data_results[key].append(value)
    for param in params[symbol]:
        data_results[param].append(getattr(strategy_results[0].params, param))

    for i, strat in enumerate(random_results):
        flattened_results = utils.get_strategy_stats(
            strat[0], flatten=True, opt=True
        )
        if flattened_results is None:
            raise RuntimeError("The strategy made no trades in the test data")
        data_results["symbol"].append(symbol)
        for key, value in flattened_results.items():
            data_results[key].append(value)
        for param in params[symbol]:
            data_results[param].append(getattr(strat[0].params, param))

        plt.plot(
            strat[0].datetimes,
            strat[0].equity_curve,
            color="lightgrey",
            alpha=0.5,
            zorder=1,
        )

    # Set the y-ticks and labels
    plt.title(f"{args.strategy} Equity Curve vs Random Exit")
    plt.xlabel("Hours")
    plt.ylabel("Portfolio Value ($USD)")

    plt.yscale("log")
    # plt.legend(fontsize='x-small')
    plt.tight_layout()
    if not os.path.exists(args.strategy_report + "/mcrandomexit/"):
        os.makedirs(args.strategy_report + "/mcrandomexit/")
    plt.savefig(
        f"{args.strategy_report}/mcrandomexit/{symbol}_mcrandomexit_curve.png"
    )
    pd.DataFrame(data_results).to_csv(
        f"{args.strategy_report}/mcrandomexit/{symbol}_results.csv",
        index=False,
    )


def main(args):
    for test in args.robustness_tests:
        if test == "vsrandom":
            vsrandom(args)
        elif test == "mcrandomentry":
            mc_randomized_entry(args)
        elif test == "mcrandomexit":
            mc_randomized_exit(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="This file can be used to generate "
        + "a robustness report AFTER generating a strategy report"
    )

    # General Arguments
    parser.add_argument(
        "--strategy-report",
        type=str,
        required=True,
        help="Path to the strategy report folder",
    )
    parser.add_argument(
        "--robustness-tests",
        choices=["vsrandom", "mcrandomentry", "mcrandomexit"],
        nargs="+",
        help="Type of robustness tests to run",
    )
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to the folder containing the data",
    )
    parser.add_argument(
        "--symbol", type=str, required=True, help="Security symbol"
    )
    parser.add_argument(
        "--strategy",
        type=str,
        required=True,
        help="Strategy to use for backtesting",
        choices=["WilliamsR", "CCI", "Stochastic", "Turnaround"],
    )

    # Test Arguments
    parser.add_argument(
        "--cash",
        type=int,
        default=100000,
        help="Starting Cash for trading",
    )
    parser.add_argument(
        "--cheat-on-open", action="store_true", help="Cheat on open"
    )
    parser.add_argument("--commission", type=float, default=0.0)
    parser.add_argument(
        "--vsrandom-itrs",
        type=int,
        default=100,
        help="Number of iterations for the vsrandom test",
    )
    parser.add_argument(
        "--mcrandom-itrs",
        type=int,
        default=100,
        help="Number of iterations for the vsrandom test",
    )

    args = parser.parse_args()

    main(args)
