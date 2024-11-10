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
from strategies import RandomStrategy, WilliamsRStrategy

def strategy_class(strategy):
  if strategy == 'WilliamsR':
    return WilliamsRStrategy
  elif strategy == 'Random':
    return RandomStrategy

def vsrandom(args):
  '''
  Compare strategy results to a random entry and exit strategy
  '''
  params = pd.read_csv(args.strategy_report+"/best_parameters.csv").set_index("symbol").to_dict(orient="index")
  for file in os.listdir(args.data):
    symbol = file.split(".")[0]
    random_results = []
    for _ in tqdm(range(args.vsrandom_itrs), desc="Running vsrandom test"):
      random_results.append(backtest(f"{args.data}/{file}", RandomStrategy, {}, args))

    plt.figure(figsize=(10, 6), dpi=300)
    plt.axhline(y=100000, color='black', linestyle='--', linewidth=2)

    for results in random_results:
      equity_curve = results[0].analyzers.cash_value.get_analysis()
      plt.plot(equity_curve, color='lightgrey', alpha=0.5)

    strategy_results = backtest(f"{args.data}/{file}", strategy_class(args.strategy), params[symbol], args)
    plt.plot(strategy_results[0].analyzers.cash_value.get_analysis(), color='red', alpha=1)

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
  strategy_results = {}
  random_results = defaultdict(list)
  for file in os.listdir(args.data):
    symbol = file.split(".")[0]
    strategy_results[symbol] = backtest(f"{args.data}/{file}", strategy, params[symbol], args)
  
  strategy.long_condition = randomized_entry
  for file in os.listdir(args.data):
    symbol = file.split(".")[0]
    for _ in tqdm(range(args.mcrandom_itrs), desc="Running mcrandom test"):
      random_results[symbol].append(backtest(f"{args.data}/{file}", strategy, params[symbol], args))

  for symbol in strategy_results:
    plt.figure(figsize=(10, 6), dpi=300)
    plt.axhline(y=100000, color='black', linestyle='--', linewidth=2)

    plt.plot(strategy_results[symbol][0].analyzers.cash_value.get_analysis(), color='red', alpha=1, zorder=3)
    for rand_result in random_results[symbol]:
      plt.plot(rand_result[0].analyzers.cash_value.get_analysis(), color='lightgrey', alpha=0.5, zorder=1)

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
  strategy_results = {}
  random_results = defaultdict(list)
  for file in os.listdir(args.data):
    symbol = file.split(".")[0]
    strategy_results[symbol] = backtest(f"{args.data}/{file}", strategy, params[symbol], args)
  
  strategy.close_condition = randomized_exit
  for file in os.listdir(args.data):
    symbol = file.split(".")[0]
    for _ in tqdm(range(args.mcrandom_itrs), desc="Running mcrandom test"):
      random_results[symbol].append(backtest(f"{args.data}/{file}", strategy, params[symbol], args))

  for symbol in strategy_results:
    plt.figure(figsize=(10, 6), dpi=300)
    plt.axhline(y=100000, color='black', linestyle='--', linewidth=2)

    plt.plot(strategy_results[symbol][0].analyzers.cash_value.get_analysis(), color='red', alpha=1, zorder=3)
    for rand_result in random_results[symbol]:
      plt.plot(rand_result[0].analyzers.cash_value.get_analysis(), color='lightgrey', alpha=0.5, zorder=1)

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
  parser.add_argument('--strategy', type=str, required=True, 
                    help="Strategy to use for backtesting", 
                    choices=['WilliamsR'])

  # Test Arguments
  parser.add_argument('--cash', type=int, default=100000, help="Starting Cash for trading")
  parser.add_argument('--cheat-on-open', action='store_true', help="Cheat on open")
  parser.add_argument('--commission', type=float, default=0.0)
  parser.add_argument("--vsrandom-itrs", type=int, default=100, help="Number of iterations for the vsrandom test")
  parser.add_argument("--mcrandom-itrs", type=int, default=100, help="Number of iterations for the vsrandom test")
  
  args = parser.parse_args()

  main(args)