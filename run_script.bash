#!/bin/bash

# Run the Python file with arguments
# /Users/briantufts/miniconda3/envs/qfaat/bin/python strategy_report.py --cash 100000 --cheat-on-open --opt-param cagr --opt-neighbors 3 --commission 0.0 --output-folder exp --strategy WilliamsR --strat-args "min_period:2 max_period:5 period_step:1 min_lowerband:-80 max_lowerband:-70 lowerband_step:5 min_upperband:-30 max_upperband:-20 upperband_step:5" --train-data Data/train --test-data Data/test
/Users/briantufts/miniconda3/envs/qfaat/bin/python robustness_report.py --cheat-on-open --strategy-report Reports/exp_2 --strategy WilliamsR --data Data/forex_data --robustness-tests mcrandomentry mcrandomexit