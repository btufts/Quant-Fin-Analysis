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
import utils
from strategies import (
    WilliamsRStrategy,
    RandomStrategy,
    CCIStrategy,
    StochasticStrategy,
    TurnaroundTuesday,
)
from analyzers import InMarketAnalyzer, CashValueAnalyzer, SortinoRatio


def opt_universe(data_path, strategy, optimization_args, args):
    data_df = pd.read_csv(data_path)
    data_df["Datetime"] = pd.to_datetime(
        data_df["Datetime"], utc=True
    ).dt.tz_convert(None)
    start_date = data_df["Datetime"].min()
    end_date = data_df["Datetime"].max()

    cerebro = bt.Cerebro(cheat_on_open=args.cheat_on_open)
    data = utils.YahooFinanceCSVData(
        dataname=data_path,
        fromdate=start_date,
        todate=end_date,
        adjclose=True,
        round=False,
    )
    cerebro.adddata(data)

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


def backtest(data_path, strategy, parameters, args):
    cerebro = bt.Cerebro(cheat_on_open=args.cheat_on_open)
    data = utils.YahooFinanceCSVData(
        dataname=data_path, adjclose=True, round=False
    )
    cerebro.adddata(data)

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


def main(args):
    output_folder = f"Reports/{args.output_folder}"
    os.makedirs(f"Reports/{args.output_folder}", exist_ok=True)

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
        **args.strat_args
    )

    opt_run = opt_universe(
        f"{args.train_data}", strategy, optimization_args, args
    )

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
            f"{args.test_data}/{symbol}_test.csv",
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Main file for generating backtesting strategy reports"
    )

    # Miscleanous arguments
    parser.add_argument(
        "--cash", type=int, default=100000, help="Starting Cash for trading"
    )
    parser.add_argument(
        "--cheat-on-open", action="store_true", help="Cheat on open"
    )
    parser.add_argument("--logging", action="store_true", help="Print Logging")
    parser.add_argument(
        "--opt-param",
        type=str,
        default="cagr",
        choices=["cagr", "sharpe", "pf"],
    )
    parser.add_argument("--opt-neighbors", type=int, default=3)
    parser.add_argument("--commission", type=float, default=0.0)
    parser.add_argument(
        "--output-folder",
        type=str,
        default="exp",
        help="Folder to save strategy report, increments if it already exists",
    )

    # Data arguments
    parser.add_argument(
        "--symbol", type=str, required=True, help="Security symbol"
    )
    parser.add_argument(
        "--train-data",
        type=str,
        required=True,
        help="Path to training data, each file"
        + " should be named symbol_train.csv",
    )
    parser.add_argument(
        "--test-data",
        type=str,
        required=True,
        help="Path to testing data, test data must include"
        + " the same symbols as train data named symbol_test.csv",
    )

    # Strategy arguments
    parser.add_argument(
        "--strategy",
        type=str,
        required=True,
        help="Strategy to use for backtesting",
        choices=["WilliamsR", "CCI", "Stochastic", "Turnaround"],
    )
    parser.add_argument(
        "--strat-args",
        type=str,
        default="",
        help="Additional arguments for the strategy wrapped"
        + ' in quotes "arg1:value1 arg2:value2"',
    )

    args = parser.parse_args()

    args.strat_args = {
        key: float(value)
        for key, value in [
            arg.split(":") for arg in args.strat_args.split(" ")
        ]
    }

    main(args)
