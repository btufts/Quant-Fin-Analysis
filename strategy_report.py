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
from strategies import WilliamsRStrategy, RandomStrategy
from analyzers import InMarketAnalyzer, CashValueAnalyzer, SortinoRatio

def opt_universe(data_path, strategy, optimization_args, args):

  data_df = pd.read_csv(data_path)
  data_df['Datetime'] = pd.to_datetime(data_df['Datetime'], utc=True).dt.tz_convert(None)
  start_date = data_df['Datetime'].min()
  end_date = data_df['Datetime'].max()

  cerebro = bt.Cerebro(cheat_on_open=args.cheat_on_open)
  data = bt.feeds.YahooFinanceCSVData(dataname=data_path, 
                                      fromdate=start_date, 
                                      todate=end_date, 
                                      adjclose=True, 
                                      round=False)
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
  cerebro.addanalyzer(bt.analyzers.TimeReturn, _name='timereturn')
  cerebro.addanalyzer(InMarketAnalyzer, _name="in_market")
  cerebro.addanalyzer(CashValueAnalyzer, _name="cash_value")
  cerebro.addanalyzer(SortinoRatio, _name='sortino')

  # Run the strategy
  runs = cerebro.run()

  return runs

def get_opt_universe_df(results, save_folder="", save=True):
  data_results = defaultdict(list)
  for symbol in results:
    runs = results[symbol]
    for strat in runs:
      returns_stats = strat[0].analyzers.returns.get_analysis()
      trade_stats = strat[0].analyzers.trade_stats.get_analysis()
      drawdown_stats = strat[0].analyzers.drawdown.get_analysis()
      in_market_stats = strat[0].analyzers.in_market.get_analysis()

      data_results["symbol"].append(symbol)
      data_results["period"].append(strat[0].params.period)
      data_results["lowerband"].append(strat[0].params.lowerband)
      data_results["upperband"].append(strat[0].params.upperband)
      data_results["total_return"].append(returns_stats['rtot'])
      data_results["cagr"].append(returns_stats['rnorm'])
      data_results["return_per_exposer"].append(returns_stats['rnorm'] / (in_market_stats["Total In-Market Bars"] / in_market_stats["Total Bars"]))
      data_results["sharpe"].append(strat[0].analyzers.sharpe.get_analysis()['sharperatio'])
      data_results["sortino"].append(strat[0].analyzers.sortino.get_analysis()['Sortino Ratio'])
      data_results["total_trades"].append(trade_stats.total.closed)
      data_results["winning_trades"].append(trade_stats.won.total)
      data_results["losing_trades"].append(trade_stats.lost.total)
      data_results["time_in_market"].append(in_market_stats["Total In-Market Bars"] / in_market_stats["Total Bars"])
      data_results["profit_factor"].append(in_market_stats["Total Gains"] / in_market_stats["Total Losses"])
      data_results["avg_gain"].append(in_market_stats["Total Percent Gain"] / trade_stats.total.closed)
      data_results["max_drawdown"].append(drawdown_stats.max.drawdown)
      data_results["max_drawdown_duration"].append(drawdown_stats.max.len)
  df = pd.DataFrame(data_results)
  if save:
    df.to_csv(f"{save_folder}/optimization_results.csv", index=False)
  return df

def get_best_parameters_df(best_params, opt_params, save_folder="", save=True):
  data_results = defaultdict(list)
  for symbol, values in best_params.items():
    data_results["symbol"].append(symbol)
    for param in opt_params.keys():
      data_results[param].append(values[param])
  df = pd.DataFrame(data_results)
  if save:
    df.to_csv(f"{save_folder}/best_parameters.csv", index=False)
  return df

def get_test_results_df(results, save_folder="", save=True):
  data_results = defaultdict(list)
  for symbol in results:
    strat = results[symbol]
    returns_stats = strat[0].analyzers.returns.get_analysis()
    trade_stats = strat[0].analyzers.trade_stats.get_analysis()
    drawdown_stats = strat[0].analyzers.drawdown.get_analysis()
    in_market_stats = strat[0].analyzers.in_market.get_analysis()

    data_results["symbol"].append(symbol)
    data_results["period"].append(strat[0].params.period)
    data_results["lowerband"].append(strat[0].params.lowerband)
    data_results["upperband"].append(strat[0].params.upperband)
    data_results["total_return"].append(returns_stats['rtot'])
    data_results["cagr"].append(returns_stats['rnorm'])
    data_results["return_per_exposer"].append(returns_stats['rnorm'] / (in_market_stats["Total In-Market Bars"] / in_market_stats["Total Bars"]))
    data_results["sharpe"].append(strat[0].analyzers.sharpe.get_analysis()['sharperatio'])
    data_results["sortino"].append(strat[0].analyzers.sortino.get_analysis()['Sortino Ratio'])
    data_results["total_trades"].append(trade_stats.total.closed)
    data_results["winning_trades"].append(trade_stats.won.total)
    data_results["losing_trades"].append(trade_stats.lost.total)
    data_results["time_in_market"].append(in_market_stats["Total In-Market Bars"] / in_market_stats["Total Bars"])
    data_results["profit_factor"].append(in_market_stats["Total Gains"] / in_market_stats["Total Losses"])
    data_results["avg_gain"].append(in_market_stats["Total Percent Gain"] / trade_stats.total.closed)
    data_results["max_drawdown"].append(drawdown_stats.max.drawdown)
    data_results["max_drawdown_duration"].append(drawdown_stats.max.len)

  df = pd.DataFrame(data_results)
  if save:
    df.to_csv(f"{save_folder}/test_results.csv", index=False)
  return df

def backtest(data_path, strategy, parameters, args):
  cerebro = bt.Cerebro(cheat_on_open=args.cheat_on_open)
  data = bt.feeds.YahooFinanceCSVData(dataname=data_path, adjclose=True, round=False)
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
  cerebro.addanalyzer(bt.analyzers.SharpeRatio, 
                    _name='sharpe',
                    timeframe=bt.TimeFrame.Minutes,  # Timeframe to match hourly data
                    compression=60,                  # Compression for hourly data (60 minutes)
                    riskfreerate=0.0)
  cerebro.addanalyzer(bt.analyzers.TimeReturn, _name='timereturn')
  cerebro.addanalyzer(InMarketAnalyzer, _name="in_market")
  cerebro.addanalyzer(CashValueAnalyzer, _name="cash_value")
  cerebro.addanalyzer(SortinoRatio, _name='sortino')
  
  results = cerebro.run()

  return results

def main(args):
  output_folder = f"Reports/{args.output_folder}"
  if not os.path.exists(output_folder):
    os.makedirs(output_folder)
  else:
    i = 1
    while os.path.exists(f"{output_folder}_{i}"):
      i += 1
    output_folder = f"{output_folder}_{i}"
    os.makedirs(output_folder)

  strategy = None
  if args.strategy == 'Random':
    strategy = RandomStrategy
  elif args.strategy == 'WilliamsR':
    strategy = WilliamsRStrategy
  else:
    exit(f"Strategy {args.strategy} not found")

  # This will build an optimization universe for each of the symbols in the training data
  optimization_args, opt_step_sizes = strategy.get_optimization_args(**args.strat_args)
  symbol_runs = {}
  for train_data in tqdm(os.listdir(args.train_data), desc="Building Optimization Universe"):
    symbol_runs[train_data.split("_")[0]] = opt_universe(f"{args.train_data}/{train_data}", strategy, optimization_args, args)
  
  opt_uni_df = get_opt_universe_df(symbol_runs, save_folder=output_folder)

  # Get the best parameters for each symbol
    # Best parameters are by symbol and by strategy
  best_params = utils.find_best_params(opt_uni_df, args.opt_param, opt_step_sizes, n=args.opt_neighbors)
  best_params_df = get_best_parameters_df(best_params, opt_step_sizes, save_folder=output_folder)

  # Backtest the best parameters for each symbol on the test data
  test_results = {}
  for symbol in tqdm(best_params_df['symbol'], desc="Backtesting Best Parameters"):
    test_results[symbol] = backtest(f'{args.test_data}/{symbol}_test.csv', strategy, best_params[symbol], args)
  
  test_results_df = get_test_results_df(test_results, save_folder=output_folder)
  
  plt.figure(figsize=(10, 6), dpi=300)
  plt.axhline(y=100000, color='black', linestyle='--', linewidth=2)

  for symbol in test_results.keys():
    equity_curve = test_results[symbol][0].analyzers.cash_value.get_analysis()
    plt.plot(equity_curve, label=f'{symbol}')

  # Set the y-ticks and labels
  plt.title(f"{args.strategy} Strategy Equity Curve, {args.opt_neighbors} neighbors, {args.opt_param} optimization")
  plt.xlabel('Hours')
  plt.ylabel('Portfolio Value ($USD)')

  plt.yscale('log')
  plt.legend(fontsize='x-small')
  plt.tight_layout()
  plt.savefig(f"{output_folder}/equity_curve.png")

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Main file for generating backtesting strategy reports")
  
  # Miscleanous arguments
  parser.add_argument('--cash', type=int, default=100000, help="Starting Cash for trading")
  parser.add_argument('--cheat-on-open', action='store_true', help="Cheat on open")
  parser.add_argument('--opt-param', type=str, default='cagr')
  parser.add_argument('--opt-neighbors', type=int, default=3)
  parser.add_argument('--commission', type=float, default=0.0)
  parser.add_argument('--output-folder', type=str, default='exp', help="Folder to save strategy report, increments if it already exists")

  # Data arguments
  parser.add_argument('--train-data', type=str, default='Data/train', help="Path to training data, each file should be named symbol_train.csv")
  parser.add_argument('--test-data', type=str, default='Data/test', help="Path to testing data, test data must include the same symbols as train data named symbol_test.csv")

  # Strategy arguments
  parser.add_argument('--strategy', type=str, required=True, 
                      help="Strategy to use for backtesting", 
                      choices=['WilliamsR'])
  parser.add_argument('--strat-args', type=str, default='', help="Additional arguments for the strategy wrapped in quotes \"arg1:value1 arg2:value2\"")
  
  args = parser.parse_args()

  args.strat_args = {key:float(value) for key,value in [arg.split(":") for arg in args.strat_args.split(" ")]}

  main(args)