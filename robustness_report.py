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
from strategy_report import backtest
from strategies import RandomStrategy, WilliamsRStrategy, CCIStrategy

def strategy_class(strategy):
  if strategy == 'WilliamsR':
    return WilliamsRStrategy
  elif strategy == 'Random':
    return RandomStrategy
  elif strategy == 'CCI':
    return CCIStrategy

def vsrandom(args):
  '''
  Compare strategy results to a random entry and exit strategy
  '''
  params = pd.read_csv(args.strategy_report+"/best_parameters.csv").set_index("symbol").to_dict(orient="index")
  
  data_results = defaultdict(list)
  symbol = args.data.split(".")[0]
  symbol = args.symbol
  random_results = []
  for _ in tqdm(range(args.vsrandom_itrs), desc="Running vsrandom test"):
    random_results.append(backtest(f"{args.data}", RandomStrategy, {}, args))

  plt.figure(figsize=(10, 6), dpi=300)
  plt.axhline(y=100000, color='black', linestyle='--', linewidth=2)

  for i, strat in enumerate(random_results):
    equity_curve = strat[0].analyzers.cash_value.get_analysis()
    returns_stats = strat[0].analyzers.returns.get_analysis()
    trade_stats = strat[0].analyzers.trade_stats.get_analysis()
    drawdown_stats = strat[0].analyzers.drawdown.get_analysis()
    in_market_stats = strat[0].analyzers.in_market.get_analysis()
    data_results["symbol"].append(f"random{i+1}")
    data_results["period"].append(-1)
    data_results["lowerband"].append(-1)
    data_results["upperband"].append(-1)
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
    
    plt.plot(equity_curve, color='lightgrey', alpha=0.5, zorder=1)

  strategy_results = backtest(f"{args.data}", strategy_class(args.strategy), params[symbol], args)
  returns_stats = strategy_results[0].analyzers.returns.get_analysis()
  trade_stats = strategy_results[0].analyzers.trade_stats.get_analysis()
  drawdown_stats = strategy_results[0].analyzers.drawdown.get_analysis()
  in_market_stats = strategy_results[0].analyzers.in_market.get_analysis()
  data_results["symbol"].append(symbol)
  data_results["period"].append(strategy_results[0].params.period)
  data_results["lowerband"].append(strategy_results[0].params.lowerband)
  data_results["upperband"].append(strategy_results[0].params.upperband)
  data_results["total_return"].append(returns_stats['rtot'])
  data_results["cagr"].append(returns_stats['rnorm'])
  data_results["return_per_exposer"].append(returns_stats['rnorm'] / (in_market_stats["Total In-Market Bars"] / in_market_stats["Total Bars"]))
  data_results["sharpe"].append(strategy_results[0].analyzers.sharpe.get_analysis()['sharperatio'])
  data_results["sortino"].append(strategy_results[0].analyzers.sortino.get_analysis()['Sortino Ratio'])
  data_results["total_trades"].append(trade_stats.total.closed)
  data_results["winning_trades"].append(trade_stats.won.total)
  data_results["losing_trades"].append(trade_stats.lost.total)
  data_results["time_in_market"].append(in_market_stats["Total In-Market Bars"] / in_market_stats["Total Bars"])
  data_results["profit_factor"].append(in_market_stats["Total Gains"] / in_market_stats["Total Losses"])
  data_results["avg_gain"].append(in_market_stats["Total Percent Gain"] / trade_stats.total.closed)
  data_results["max_drawdown"].append(drawdown_stats.max.drawdown)
  data_results["max_drawdown_duration"].append(drawdown_stats.max.len)
  plt.plot(strategy_results[0].analyzers.cash_value.get_analysis(), color='red', alpha=1, zorder=3)

  # Set the y-ticks and labels
  plt.title(f"{args.strategy} Equity Curve vs Random")
  plt.xlabel('Hours')
  plt.ylabel('Portfolio Value ($USD)')

  plt.yscale('log')
  # plt.legend(fontsize='x-small')
  plt.tight_layout()
  if not os.path.exists(args.strategy_report+"/vsrandom/"):
    os.makedirs(args.strategy_report+"/vsrandom/")
  plt.savefig(f"{args.strategy_report}/vsrandom/{symbol}_vsrandom_curve.png")
  pd.DataFrame(data_results).to_csv(f"{args.strategy_report}/vsrandom/{symbol}_results.csv", index=False)

def mc_randomized_entry(args):
  '''
  Randomize the entry of the strategy to see if the exit is robust
  '''
  params = pd.read_csv(args.strategy_report+"/best_parameters.csv").set_index("symbol").to_dict(orient="index")
  strategy = strategy_class(args.strategy)
  
  def randomized_entry(self, *args, **kwargs):
    action = random.randint(0,2)
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
    random_results.append(backtest(f"{args.data}", strategy, params[symbol], args))

  data_results = defaultdict(list)
  plt.figure(figsize=(10, 6), dpi=300)
  plt.axhline(y=100000, color='black', linestyle='--', linewidth=2)

  plt.plot(strategy_results[0].analyzers.cash_value.get_analysis(), color='red', alpha=1, zorder=3)
  returns_stats = strategy_results[0].analyzers.returns.get_analysis()
  trade_stats = strategy_results[0].analyzers.trade_stats.get_analysis()
  drawdown_stats = strategy_results[0].analyzers.drawdown.get_analysis()
  in_market_stats = strategy_results[0].analyzers.in_market.get_analysis()
  data_results["symbol"].append(symbol)
  data_results["period"].append(strategy_results[0].params.period)
  data_results["lowerband"].append(strategy_results[0].params.lowerband)
  data_results["upperband"].append(strategy_results[0].params.upperband)
  data_results["total_return"].append(returns_stats['rtot'])
  data_results["cagr"].append(returns_stats['rnorm'])
  data_results["return_per_exposer"].append(returns_stats['rnorm'] / (in_market_stats["Total In-Market Bars"] / in_market_stats["Total Bars"]))
  data_results["sharpe"].append(strategy_results[0].analyzers.sharpe.get_analysis()['sharperatio'])
  data_results["sortino"].append(strategy_results[0].analyzers.sortino.get_analysis()['Sortino Ratio'])
  data_results["total_trades"].append(trade_stats.total.closed)
  data_results["winning_trades"].append(trade_stats.won.total)
  data_results["losing_trades"].append(trade_stats.lost.total)
  data_results["time_in_market"].append(in_market_stats["Total In-Market Bars"] / in_market_stats["Total Bars"])
  data_results["profit_factor"].append(in_market_stats["Total Gains"] / in_market_stats["Total Losses"])
  data_results["avg_gain"].append(in_market_stats["Total Percent Gain"] / trade_stats.total.closed)
  data_results["max_drawdown"].append(drawdown_stats.max.drawdown)
  data_results["max_drawdown_duration"].append(drawdown_stats.max.len)

  for i, strat in enumerate(random_results):
    returns_stats = strat[0].analyzers.returns.get_analysis()
    trade_stats = strat[0].analyzers.trade_stats.get_analysis()
    drawdown_stats = strat[0].analyzers.drawdown.get_analysis()
    in_market_stats = strat[0].analyzers.in_market.get_analysis()
    data_results["symbol"].append(f"random{i+1}")
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
    plt.plot(strat[0].analyzers.cash_value.get_analysis(), color='lightgrey', alpha=0.5, zorder=1)

  # Set the y-ticks and labels
  plt.title(f"{args.strategy} Equity Curve vs Random Entry")
  plt.xlabel('Hours')
  plt.ylabel('Portfolio Value ($USD)')

  plt.yscale('log')
  # plt.legend(fontsize='x-small')
  plt.tight_layout()
  if not os.path.exists(args.strategy_report+"/mcrandomentry/"):
    os.makedirs(args.strategy_report+"/mcrandomentry/")
  plt.savefig(f"{args.strategy_report}/mcrandomentry/{symbol}_mcrandomentry_curve.png")
  pd.DataFrame(data_results).to_csv(f"{args.strategy_report}/mcrandomentry/{symbol}_results.csv", index=False)


def mc_randomized_exit(args):
  '''
  Randomize the exit of the strategy to see if the entry is robust
  '''
  params = pd.read_csv(args.strategy_report+"/best_parameters.csv").set_index("symbol").to_dict(orient="index")
  strategy = strategy_class(args.strategy)
  
  def randomized_exit(self, *args, **kwargs):
    action = random.randint(0,2)
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
    random_results.append(backtest(f"{args.data}", strategy, params[symbol], args))

  data_results = defaultdict(list)
  plt.figure(figsize=(10, 6), dpi=300)
  plt.axhline(y=100000, color='black', linestyle='--', linewidth=2)

  plt.plot(strategy_results[0].analyzers.cash_value.get_analysis(), color='red', alpha=1, zorder=3)
  returns_stats = strategy_results[0].analyzers.returns.get_analysis()
  trade_stats = strategy_results[0].analyzers.trade_stats.get_analysis()
  drawdown_stats = strategy_results[0].analyzers.drawdown.get_analysis()
  in_market_stats = strategy_results[0].analyzers.in_market.get_analysis()
  data_results["symbol"].append(symbol)
  data_results["period"].append(strategy_results[0].params.period)
  data_results["lowerband"].append(strategy_results[0].params.lowerband)
  data_results["upperband"].append(strategy_results[0].params.upperband)
  data_results["total_return"].append(returns_stats['rtot'])
  data_results["cagr"].append(returns_stats['rnorm'])
  data_results["return_per_exposer"].append(returns_stats['rnorm'] / (in_market_stats["Total In-Market Bars"] / in_market_stats["Total Bars"]))
  data_results["sharpe"].append(strategy_results[0].analyzers.sharpe.get_analysis()['sharperatio'])
  data_results["sortino"].append(strategy_results[0].analyzers.sortino.get_analysis()['Sortino Ratio'])
  data_results["total_trades"].append(trade_stats.total.closed)
  data_results["winning_trades"].append(trade_stats.won.total)
  data_results["losing_trades"].append(trade_stats.lost.total)
  data_results["time_in_market"].append(in_market_stats["Total In-Market Bars"] / in_market_stats["Total Bars"])
  data_results["profit_factor"].append(in_market_stats["Total Gains"] / in_market_stats["Total Losses"])
  data_results["avg_gain"].append(in_market_stats["Total Percent Gain"] / trade_stats.total.closed)
  data_results["max_drawdown"].append(drawdown_stats.max.drawdown)
  data_results["max_drawdown_duration"].append(drawdown_stats.max.len)

  for i, strat in enumerate(random_results):
    returns_stats = strat[0].analyzers.returns.get_analysis()
    trade_stats = strat[0].analyzers.trade_stats.get_analysis()
    drawdown_stats = strat[0].analyzers.drawdown.get_analysis()
    in_market_stats = strat[0].analyzers.in_market.get_analysis()
    data_results["symbol"].append(f"random{i+1}")
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
    plt.plot(strat[0].analyzers.cash_value.get_analysis(), color='lightgrey', alpha=0.5, zorder=1)

  # Set the y-ticks and labels
  plt.title(f"{args.strategy} Equity Curve vs Random Exit")
  plt.xlabel('Hours')
  plt.ylabel('Portfolio Value ($USD)')

  plt.yscale('log')
  # plt.legend(fontsize='x-small')
  plt.tight_layout()
  if not os.path.exists(args.strategy_report+"/mcrandomexit/"):
    os.makedirs(args.strategy_report+"/mcrandomexit/")
  plt.savefig(f"{args.strategy_report}/mcrandomexit/{symbol}_mcrandomexit_curve.png")
  pd.DataFrame(data_results).to_csv(f"{args.strategy_report}/mcrandomexit/{symbol}_results.csv", index=False)

def main(args):
  for test in args.robustness_tests:
    if test == 'vsrandom':
      vsrandom(args)
    elif test == 'mcrandomentry':
      mc_randomized_entry(args)
    elif test == 'mcrandomexit':
      mc_randomized_exit(args)

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="This file can be used to generate a robustness report AFTER generating a strategy report")
  
  # General Arguments
  parser.add_argument("--strategy-report", type=str, required=True, help="Path to the strategy report folder")
  parser.add_argument("--robustness-tests", choices=['vsrandom', 'mcrandomentry', 'mcrandomexit'], nargs='+', help="Type of robustness tests to run")  
  parser.add_argument("--data", type=str, required=True, help="Path to the folder containing the data")
  parser.add_argument('--symbol', type=str, required=True, help='Security symbol')
  parser.add_argument('--strategy', type=str, required=True, 
                    help="Strategy to use for backtesting", 
                    choices=['WilliamsR', 'CCI'])

  # Test Arguments
  parser.add_argument('--cash', type=int, default=100000, help="Starting Cash for trading")
  parser.add_argument('--cheat-on-open', action='store_true', help="Cheat on open")
  parser.add_argument('--commission', type=float, default=0.0)
  parser.add_argument("--vsrandom-itrs", type=int, default=100, help="Number of iterations for the vsrandom test")
  parser.add_argument("--mcrandom-itrs", type=int, default=100, help="Number of iterations for the vsrandom test")
  
  args = parser.parse_args()

  main(args)