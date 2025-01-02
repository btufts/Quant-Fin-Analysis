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


def vsrandom(args, output_folder):
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
        pd.read_csv(args.optimize_result + "/best_parameters.csv")
        .set_index("symbol")
        .to_dict(orient="index")
    )

    data_results = defaultdict(list)
    symbol = args.data.split(".")[0]
    symbol = args.symbol
    random_results = []
    df = pd.read_csv(f"{args.data}")
    for _ in tqdm(range(args.vsrandom_itrs), desc="Running vsrandom test"):
        random_results.append(backtest(df, RandomStrategy, {}, args))

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
        df, strategy_class(args.strategy), params[symbol], args
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
    os.makedirs(f"{output_folder}/vsrandom/", exist_ok=True)
    plt.savefig(f"{output_folder}/vsrandom/{symbol}_vsrandom_curve.png")
    pd.DataFrame(data_results).to_csv(
        f"{output_folder}/vsrandom/{symbol}_results.csv",
        index=False,
    )


def mc_randomized_entry(args, output_folder):
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
        pd.read_csv(args.optimize_result + "/best_parameters.csv")
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
    df = pd.read_csv(f"{args.data}")
    strategy_results = backtest(df, strategy, params[symbol], args)

    strategy.long_condition = randomized_entry
    for _ in tqdm(range(args.mcrandom_itrs), desc="Running mcrandom test"):
        random_results.append(backtest(df, strategy, params[symbol], args))

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
    os.makedirs(
        f"{output_folder}/mcrandomentry/",
        exist_ok=True,
    )
    plt.savefig(
        f"{output_folder}/mcrandomentry" + f"/{symbol}_mcrandomentry_curve.png"
    )
    pd.DataFrame(data_results).to_csv(
        f"{output_folder}/mcrandomentry/{symbol}_results.csv",
        index=False,
    )


def mc_randomized_exit(args, output_folder):
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
        pd.read_csv(args.optimize_result + "/best_parameters.csv")
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
    df = pd.read_csv(f"{args.data}")
    strategy_results = backtest(df, strategy, params[symbol], args)

    strategy.long_condition = randomized_exit
    for _ in tqdm(range(args.mcrandom_itrs), desc="Running mcrandom test"):
        random_results.append(backtest(df, strategy, params[symbol], args))

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
    os.makedirs(
        f"{output_folder}/mcrandomexit/",
        exist_ok=True,
    )
    plt.savefig(
        f"{output_folder}/mcrandomexit/{symbol}_mcrandomexit_curve.png"
    )
    pd.DataFrame(data_results).to_csv(
        f"{output_folder}/mcrandomexit/{symbol}_results.csv",
        index=False,
    )


def main(args, root_folder):
    args.optimize_result = f"{root_folder}/{args.optimize_result}"
    output_folder = f"{root_folder}/{args.optimize_result}/{args.name}"
    for test in args.tests:
        if test == "vsrandom":
            vsrandom(args, output_folder)
        elif test == "mcrandomentry":
            mc_randomized_entry(args, output_folder)
        elif test == "mcrandomexit":
            mc_randomized_exit(args, output_folder)
