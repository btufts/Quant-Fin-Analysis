# System Imports
import os
from collections import defaultdict
import os.path
import argparse

# Third Party Imports
import pandas as pd
from tqdm import tqdm
import backtrader as bt
import backtrader.analyzers as btanalyzers
import matplotlib.pyplot as plt

# Local Imports
import utils.utils as utils
from strategies import (
    WilliamsRStrategy,
    RandomStrategy,
    CCIStrategy,
    StochasticStrategy,
    TurnaroundTuesday,
)
from analyzers import InMarketAnalyzer, CashValueAnalyzer, SortinoRatio


def get_train_test_split(df, test_size=0.2):
    datetime_column = "Datetime"
    df[datetime_column] = pd.to_datetime(df[datetime_column])

    # Sort the dataframe by the datetime column
    df = df.sort_values(by=datetime_column)

    # Calculate the split index for the test size
    split_index = int(len(df) * (1 - test_size))

    # Split the dataframe
    train_df = df.iloc[:split_index]
    test_df = df.iloc[split_index:]

    return train_df, test_df


def opt_universe(
    df: pd.DataFrame,
    strategy: bt.Strategy,
    optimization_args: dict[str, int],
    args,
):

    start_date = df["Datetime"].min()
    end_date = df["Datetime"].max()

    cerebro = bt.Cerebro(cheat_on_open=args.cheat_on_open)

    datafeed = utils.PandasDataFeed(
        dataframe=df,
        dtformat="%Y-%m-%d %H:%M:%S",
        use_adjusted_close=True,
        timeframe=bt.TimeFrame.Minutes,
        compression=60,
    )
    cerebro.adddata(datafeed)

    # Add the WilliamsR strategy
    cerebro.optstrategy(strategy, **optimization_args)
    # cerebro.optstrategy(WilliamsRStrategy, **args.strat_args)
    cerebro.broker.setcommission(commission=args.commission)

    # Set initial capital and broker settings
    # This code was assited using Codeium autocomplete
    cerebro.broker.setcash(args.cash)
    cerebro.addsizer(bt.sizers.PercentSizer, percents=100)
    cerebro.addanalyzer(btanalyzers.TradeAnalyzer, _name="trade_stats")
    cerebro.addanalyzer(btanalyzers.DrawDown, _name="drawdown")
    cerebro.addanalyzer(btanalyzers.Returns, _name="returns")
    cerebro.addanalyzer(btanalyzers.SharpeRatio, _name="sharpe")
    cerebro.addanalyzer(bt.analyzers.TimeReturn, _name="timereturn")
    cerebro.addanalyzer(InMarketAnalyzer, _name="in_market")
    cerebro.addanalyzer(CashValueAnalyzer, _name="cash_value")
    cerebro.addanalyzer(SortinoRatio, _name="sortino")

    # Run the strategy
    runs = cerebro.run()

    return runs


def get_opt_universe_df(
    results, symbol, opt_step_sizes, save_folder="", save=True
):
    """
    Takes the results of a backtrader optimization run and returns a pandas
    DataFrame with the results. If the `save` parameter is True, the DataFrame
    is saved to a CSV file in the `save_folder` directory.

    Parameters
    ----------
    results : list
        The results of a backtrader optimization run.
    symbol : str
        The symbol of the security that was backtested.
    opt_step_sizes : dict
        A dictionary of the step sizes for the optimization parameters.
    save_folder : str, default ""
        The directory to which the results should be saved. If the directory
        does not exist, it will be created.
    save : bool, default True
        Whether or not to save the results to a CSV file.

    Returns
    -------
    pd.DataFrame
        A DataFrame with the results of the optimization run.
    """
    data_results = defaultdict(list)
    for strat in results:
        flattened_results = utils.get_strategy_stats(
            strat[0], flatten=True, opt=True
        )
        if flattened_results:
            data_results["symbol"].append(symbol)
            for key, value in flattened_results.items():
                data_results[key].append(value)
            for param in opt_step_sizes.keys():
                data_results[param].append(getattr(strat[0].params, param))

    df = pd.DataFrame(data_results)

    if save:
        if os.path.exists(f"{save_folder}/optimization_results.csv"):
            existing_df = pd.read_csv(
                f"{save_folder}/optimization_results.csv"
            )
            merge_columns = ["symbol"]
            merge_columns.extend([param for param in opt_step_sizes.keys()])
            updated_df = pd.concat(
                [
                    df,
                    existing_df[
                        ~existing_df[merge_columns]
                        .apply(tuple, 1)
                        .isin(df[merge_columns].apply(tuple, 1))
                    ],
                ]
            )
            updated_df.to_csv(
                f"{save_folder}/optimization_results.csv", index=False
            )
        else:
            df.to_csv(f"{save_folder}/optimization_results.csv", index=False)

    return df


def get_best_parameters_df(
    best_params, opt_params, symbol, save_folder="", save=True
):
    """
    Takes in best parameters and optimization parameters, and writes them to a
    pandas dataframe.
    The dataframe is then saved to a csv file in the specified folder, or
    appended to an existing csv file if it already exists.

    Parameters
    ----------
    best_params : dict
        A dictionary of the best parameters found, with parameter names as
        keys.
    opt_params : dict
        A dictionary of the optimization parameters used, with parameter names
        as keys.
    symbol : str
        The symbol of the instrument that was optimized.
    save_folder : str
        The folder where the csv file should be saved. If empty, the file is
        not saved.
    save : bool
        If True, the dataframe is saved to a csv file. If False, the dataframe
        is returned but not saved.

    Returns
    -------
    pd.DataFrame
        A pandas dataframe with the best parameters and symbol.
        If the dataframe is saved, it is returned.
        If the dataframe is not saved, it is returned.
    """
    data_results = defaultdict(list)
    data_results["symbol"].append(symbol)

    for param in opt_params.keys():
        data_results[param].append(best_params[param])
    df = pd.DataFrame(data_results)
    if save:
        if os.path.exists(f"{save_folder}/best_parameters.csv"):
            existing_df = pd.read_csv(f"{save_folder}/best_parameters.csv")
            merge_columns = ["symbol"]
            updated_df = pd.concat(
                [
                    df,
                    existing_df[
                        ~existing_df[merge_columns]
                        .apply(tuple, 1)
                        .isin(df[merge_columns].apply(tuple, 1))
                    ],
                ]
            )
            updated_df.to_csv(
                f"{save_folder}/best_parameters.csv", index=False
            )
            return updated_df
        else:
            df.to_csv(f"{save_folder}/best_parameters.csv", index=False)
            return df


def get_test_results_df(results, opt_step_sizes, save_folder="", save=True):
    """
    This function takes a dictionary of backtrader results and a dictionary
    of optimization step sizes, and returns a pandas
    DataFrame with the test results. The dataframe is saved
    to a csv file if save is True. If the file already exists, the dataframe
    is merged with the existing dataframe.

    Parameters
    ----------
    results : dict
        A dictionary of backtrader results. The keys are the symbols and the
        values are a list of a backtrader Strategy object and the name of the
        strategy.
    opt_step_sizes : dict
        A dictionary of optimization step sizes. The keys are the parameter
        names and the values are the step sizes.
    save_folder : str
        The folder where the csv file should be saved. If empty, the file is
        not saved.
    save : bool
        If True, the dataframe is saved to a csv file. If False, the dataframe
        is returned but
        not saved.

    Returns
    -------
    pd.DataFrame
        A pandas dataframe with the test results. If the dataframe is saved,
        it is returned. If the dataframe is not saved, it is returned.
    """

    data_results = defaultdict(list)
    for symbol in results:
        strat = results[symbol]
        flattened_results = utils.get_strategy_stats(
            strat[0], flatten=True, opt=True
        )
        if flattened_results is None:
            raise RuntimeError("The strategy made no trades in the test data")
        data_results["symbol"].append(symbol)
        for key, value in flattened_results.items():
            data_results[key].append(value)
        for param in opt_step_sizes.keys():
            data_results[param].append(getattr(strat[0].params, param))

    df = pd.DataFrame(data_results)

    if save:
        df.to_csv(f"{save_folder}/test_results.csv", index=False)
    return df


def backtest(
    df: pd.DataFrame, strategy: bt.Strategy, parameters: dict[str, int], args
):
    cerebro = bt.Cerebro(cheat_on_open=args.cheat_on_open)

    datafeed = utils.PandasDataFeed(
        dataframe=df,
        dtformat="%Y-%m-%d %H:%M:%S",
        use_adjusted_close=True,
        timeframe=bt.TimeFrame.Minutes,
        compression=60,
    )

    cerebro.adddata(datafeed)

    # Add the WilliamsR strategy
    cerebro.addstrategy(strategy, **parameters)
    cerebro.broker.setcommission(commission=args.commission)

    # Set initial capital and broker settings
    # This code was assited using Codeium autocomplete
    cerebro.broker.setcash(args.cash)
    cerebro.addsizer(bt.sizers.PercentSizer, percents=100)
    cerebro.addanalyzer(btanalyzers.TradeAnalyzer, _name="trade_stats")
    cerebro.addanalyzer(btanalyzers.DrawDown, _name="drawdown")
    cerebro.addanalyzer(btanalyzers.Returns, _name="returns")
    cerebro.addanalyzer(
        bt.analyzers.SharpeRatio,
        _name="sharpe",
        timeframe=bt.TimeFrame.Minutes,  # Timeframe to match hourly data
        compression=60,  # Compression for hourly data (60 minutes)
        riskfreerate=0.01,
    )
    cerebro.addanalyzer(bt.analyzers.TimeReturn, _name="timereturn")
    cerebro.addanalyzer(InMarketAnalyzer, _name="in_market")
    cerebro.addanalyzer(CashValueAnalyzer, _name="cash_value")
    cerebro.addanalyzer(SortinoRatio, _name="sortino")

    results = cerebro.run()

    return results


def main(args, root_folder):
    output_folder = f"{root_folder}/{args.name}"
    os.makedirs(f"{root_folder}/{args.name}", exist_ok=True)

    strategy = None
    if args.strategy == "Random":
        strategy = RandomStrategy
    elif args.strategy == "WilliamsR":
        strategy = WilliamsRStrategy
    elif args.strategy == "CCI":
        strategy = CCIStrategy
    elif args.strategy == "Stochastic":
        strategy = StochasticStrategy
    elif args.strategy == "Turnaround":
        strategy = TurnaroundTuesday
    else:
        raise ValueError(f"Strategy {args.strategy} not found")

    # This will build an optimization
    # universe for each of the symbols in the training data
    optimization_args, opt_step_sizes = strategy.get_optimization_args(
        **args.parameters
    )

    data_df = pd.read_csv(args.data)

    train_data, test_data = get_train_test_split(data_df)

    opt_run = opt_universe(train_data, strategy, optimization_args, args)

    opt_uni_df = get_opt_universe_df(
        opt_run, args.symbol, opt_step_sizes, save_folder=output_folder
    )

    # Get the best parameters for each symbol
    # Best parameters are by symbol and by strategy
    if args.opt_param == "cagr":
        opt_param = "returns.cagr"
    elif args.opt_param == "sharpe":
        opt_param = "returns.sharpe_ratio"
    elif args.opt_param == "pf":
        opt_param = "profit_factor"
    else:
        raise ValueError(f"Optimization parameter {args.opt_param} not found")

    best_params = utils.find_best_params(
        opt_uni_df, opt_param, opt_step_sizes, n=args.opt_neighbors
    )
    best_params_df = get_best_parameters_df(
        best_params, opt_step_sizes, args.symbol, save_folder=output_folder
    )

    # Backtest the best parameters for each
    # symbol on the test data to get graph
    test_results = {}
    best_params_dict = best_params_df.set_index("symbol").to_dict(
        orient="index"
    )
    for symbol in tqdm(
        best_params_df["symbol"], desc="Backtesting Best Parameters"
    ):
        test_results[symbol] = backtest(
            test_data,
            strategy,
            best_params_dict[symbol],
            args,
        )

    get_test_results_df(
        test_results, opt_step_sizes, save_folder=output_folder
    )

    plt.figure(figsize=(10, 6), dpi=100)
    plt.axhline(y=100000, color="black", linestyle="--", linewidth=2)

    for symbol in test_results.keys():
        equity_curve = test_results[symbol][0].equity_curve
        datetimes = test_results[symbol][0].datetimes
        plt.plot(datetimes, equity_curve, label=f"{symbol}")

    # Set the y-ticks and labels
    plt.title(
        f"{args.strategy} Strategy Equity Curve,"
        + f" {args.opt_neighbors} neighbors, {args.opt_param} optimization"
    )
    plt.xlabel("Datetime")
    plt.ylabel("Portfolio Value ($USD)")

    plt.yscale("log")
    plt.legend(fontsize="x-small")
    plt.tight_layout()
    plt.savefig(f"{output_folder}/equity_curve.png")
