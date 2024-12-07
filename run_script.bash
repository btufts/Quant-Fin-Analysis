#!/bin/bash

# Run the Python file with arguments
# python strategy_report.py --cash 100000 --cheat-on-open --opt-param cagr --opt-neighbors 3 --commission 0.0 --output-folder exp --strategy WilliamsR --strat-args "min_period:2 max_period:5 period_step:1 min_lowerband:-80 max_lowerband:-70 lowerband_step:5 min_upperband:-30 max_upperband:-20 upperband_step:5" --train-data Data/train --test-data Data/test
# python robustness_report.py --cheat-on-open --strategy-report Reports/exp_2 --strategy WilliamsR --data Data/forex_data --robustness-tests mcrandomentry mcrandomexit vsrandom
# python strategy_report.py --cash 100000 --cheat-on-open --opt-param cagr --opt-neighbors 3 --commission 0.0 --output-folder exp --strategy CCI --strat-args "min_period:10 max_period:15 period_step:1 min_lowerband:-100 max_lowerband:-90 lowerband_step:5 min_upperband:90 max_upperband:100 upperband_step:5" --train-data Data/train --test-data Data/test
# python robustness_report.py --cheat-on-open --strategy-report Reports/exp_9 --strategy CCI --data Data/forex_data --robustness-tests vsrandom

# python strategy_report.py --cash 100000 --cheat-on-open --opt-param sharpe --opt-neighbors 5 --commission 0.0 --strategy CCI --strat-args "min_period:10 max_period:15 period_step:1 min_lowerband:-100 max_lowerband:-80 lowerband_step:5 min_upperband:80 max_upperband:100 upperband_step:5" --train-data Data/train/usdaud_train.csv --test-data Data/test --output-folder ccisharpe --symbol usdaud
# python strategy_report.py --cash 100000 --cheat-on-open --opt-param sharpe --opt-neighbors 5 --commission 0.0 --strategy CCI --strat-args "min_period:10 max_period:15 period_step:1 min_lowerband:-100 max_lowerband:-80 lowerband_step:5 min_upperband:80 max_upperband:100 upperband_step:5" --train-data Data/train/usdcad_train.csv --test-data Data/test --output-folder ccisharpe --symbol usdcad
# python strategy_report.py --cash 100000 --cheat-on-open --opt-param sharpe --opt-neighbors 5 --commission 0.0 --strategy CCI --strat-args "min_period:10 max_period:15 period_step:1 min_lowerband:-100 max_lowerband:-80 lowerband_step:5 min_upperband:80 max_upperband:100 upperband_step:5" --train-data Data/train/usdchf_train.csv --test-data Data/test --output-folder ccisharpe --symbol usdchf
# python strategy_report.py --cash 100000 --cheat-on-open --opt-param sharpe --opt-neighbors 5 --commission 0.0 --strategy CCI --strat-args "min_period:10 max_period:15 period_step:1 min_lowerband:-100 max_lowerband:-80 lowerband_step:5 min_upperband:80 max_upperband:100 upperband_step:5" --train-data Data/train/usdeur_train.csv --test-data Data/test --output-folder ccisharpe --symbol usdeur
# python strategy_report.py --cash 100000 --cheat-on-open --opt-param sharpe --opt-neighbors 5 --commission 0.0 --strategy CCI --strat-args "min_period:10 max_period:15 period_step:1 min_lowerband:-100 max_lowerband:-80 lowerband_step:5 min_upperband:80 max_upperband:100 upperband_step:5" --train-data Data/train/usdgbp_train.csv --test-data Data/test --output-folder ccisharpe --symbol usdgbp
# python strategy_report.py --cash 100000 --cheat-on-open --opt-param sharpe --opt-neighbors 5 --commission 0.0 --strategy CCI --strat-args "min_period:10 max_period:15 period_step:1 min_lowerband:-100 max_lowerband:-80 lowerband_step:5 min_upperband:80 max_upperband:100 upperband_step:5" --train-data Data/train/usdhkd_train.csv --test-data Data/test --output-folder ccisharpe --symbol usdhkd
# python strategy_report.py --cash 100000 --cheat-on-open --opt-param sharpe --opt-neighbors 5 --commission 0.0 --strategy CCI --strat-args "min_period:10 max_period:15 period_step:1 min_lowerband:-100 max_lowerband:-80 lowerband_step:5 min_upperband:80 max_upperband:100 upperband_step:5" --train-data Data/train/usdjpy_train.csv --test-data Data/test --output-folder ccisharpe --symbol usdjpy
# python strategy_report.py --cash 100000 --cheat-on-open --opt-param sharpe --opt-neighbors 5 --commission 0.0 --strategy CCI --strat-args "min_period:10 max_period:15 period_step:1 min_lowerband:-100 max_lowerband:-80 lowerband_step:5 min_upperband:80 max_upperband:100 upperband_step:5" --train-data Data/train/usdmxn_train.csv --test-data Data/test --output-folder ccisharpe --symbol usdmxn
# python strategy_report.py --cash 100000 --cheat-on-open --opt-param sharpe --opt-neighbors 5 --commission 0.0 --strategy CCI --strat-args "min_period:10 max_period:15 period_step:1 min_lowerband:-100 max_lowerband:-80 lowerband_step:5 min_upperband:80 max_upperband:100 upperband_step:5" --train-data Data/train/usdnzd_train.csv --test-data Data/test --output-folder ccisharpe --symbol usdnzd
# python strategy_report.py --cash 100000 --cheat-on-open --opt-param sharpe --opt-neighbors 5 --commission 0.0 --strategy CCI --strat-args "min_period:10 max_period:15 period_step:1 min_lowerband:-100 max_lowerband:-80 lowerband_step:5 min_upperband:80 max_upperband:100 upperband_step:5" --train-data Data/train/usdsek_train.csv --test-data Data/test --output-folder ccisharpe --symbol usdsek

python robustness_report.py --cheat-on-open --strategy-report Reports/ccicagr --strategy CCI --data Data/forex_data/usdaud.csv --symbol usdaud --robustness-tests mcrandomentry mcrandomexit
python robustness_report.py --cheat-on-open --strategy-report Reports/ccicagr --strategy CCI --data Data/forex_data/usdcad.csv --symbol usdcad --robustness-tests mcrandomentry mcrandomexit
python robustness_report.py --cheat-on-open --strategy-report Reports/ccicagr --strategy CCI --data Data/forex_data/usdchf.csv --symbol usdchf --robustness-tests mcrandomentry mcrandomexit
python robustness_report.py --cheat-on-open --strategy-report Reports/ccicagr --strategy CCI --data Data/forex_data/usdeur.csv --symbol usdeur --robustness-tests mcrandomentry mcrandomexit
python robustness_report.py --cheat-on-open --strategy-report Reports/ccicagr --strategy CCI --data Data/forex_data/usdgbp.csv --symbol usdgbp --robustness-tests mcrandomentry mcrandomexit
python robustness_report.py --cheat-on-open --strategy-report Reports/ccicagr --strategy CCI --data Data/forex_data/usdhkd.csv --symbol usdhkd --robustness-tests mcrandomentry mcrandomexit
python robustness_report.py --cheat-on-open --strategy-report Reports/ccicagr --strategy CCI --data Data/forex_data/usdjpy.csv --symbol usdjpy --robustness-tests mcrandomentry mcrandomexit
python robustness_report.py --cheat-on-open --strategy-report Reports/ccicagr --strategy CCI --data Data/forex_data/usdmxn.csv --symbol usdmxn --robustness-tests mcrandomentry mcrandomexit
python robustness_report.py --cheat-on-open --strategy-report Reports/ccicagr --strategy CCI --data Data/forex_data/usdnzd.csv --symbol usdnzd --robustness-tests mcrandomentry mcrandomexit
python robustness_report.py --cheat-on-open --strategy-report Reports/ccicagr --strategy CCI --data Data/forex_data/usdsek.csv --symbol usdsek --robustness-tests mcrandomentry mcrandomexit

python robustness_report.py --cheat-on-open --strategy-report Reports/ccisharpe --strategy CCI --data Data/forex_data/usdaud.csv --symbol usdaud --robustness-tests mcrandomentry mcrandomexit
python robustness_report.py --cheat-on-open --strategy-report Reports/ccisharpe --strategy CCI --data Data/forex_data/usdcad.csv --symbol usdcad --robustness-tests mcrandomentry mcrandomexit
python robustness_report.py --cheat-on-open --strategy-report Reports/ccisharpe --strategy CCI --data Data/forex_data/usdchf.csv --symbol usdchf --robustness-tests mcrandomentry mcrandomexit
python robustness_report.py --cheat-on-open --strategy-report Reports/ccisharpe --strategy CCI --data Data/forex_data/usdeur.csv --symbol usdeur --robustness-tests mcrandomentry mcrandomexit
python robustness_report.py --cheat-on-open --strategy-report Reports/ccisharpe --strategy CCI --data Data/forex_data/usdgbp.csv --symbol usdgbp --robustness-tests mcrandomentry mcrandomexit
python robustness_report.py --cheat-on-open --strategy-report Reports/ccisharpe --strategy CCI --data Data/forex_data/usdhkd.csv --symbol usdhkd --robustness-tests mcrandomentry mcrandomexit
python robustness_report.py --cheat-on-open --strategy-report Reports/ccisharpe --strategy CCI --data Data/forex_data/usdjpy.csv --symbol usdjpy --robustness-tests mcrandomentry mcrandomexit
python robustness_report.py --cheat-on-open --strategy-report Reports/ccisharpe --strategy CCI --data Data/forex_data/usdmxn.csv --symbol usdmxn --robustness-tests mcrandomentry mcrandomexit
python robustness_report.py --cheat-on-open --strategy-report Reports/ccisharpe --strategy CCI --data Data/forex_data/usdnzd.csv --symbol usdnzd --robustness-tests mcrandomentry mcrandomexit
python robustness_report.py --cheat-on-open --strategy-report Reports/ccisharpe --strategy CCI --data Data/forex_data/usdsek.csv --symbol usdsek --robustness-tests mcrandomentry mcrandomexit

python robustness_report.py --cheat-on-open --strategy-report Reports/williamscagr --strategy WilliamsR --data Data/forex_data/usdaud.csv --symbol usdaud --robustness-tests mcrandomentry mcrandomexit
python robustness_report.py --cheat-on-open --strategy-report Reports/williamscagr --strategy WilliamsR --data Data/forex_data/usdcad.csv --symbol usdcad --robustness-tests mcrandomentry mcrandomexit
python robustness_report.py --cheat-on-open --strategy-report Reports/williamscagr --strategy WilliamsR --data Data/forex_data/usdchf.csv --symbol usdchf --robustness-tests mcrandomentry mcrandomexit
python robustness_report.py --cheat-on-open --strategy-report Reports/williamscagr --strategy WilliamsR --data Data/forex_data/usdeur.csv --symbol usdeur --robustness-tests mcrandomentry mcrandomexit
python robustness_report.py --cheat-on-open --strategy-report Reports/williamscagr --strategy WilliamsR --data Data/forex_data/usdgbp.csv --symbol usdgbp --robustness-tests mcrandomentry mcrandomexit
python robustness_report.py --cheat-on-open --strategy-report Reports/williamscagr --strategy WilliamsR --data Data/forex_data/usdhkd.csv --symbol usdhkd --robustness-tests mcrandomentry mcrandomexit
python robustness_report.py --cheat-on-open --strategy-report Reports/williamscagr --strategy WilliamsR --data Data/forex_data/usdjpy.csv --symbol usdjpy --robustness-tests mcrandomentry mcrandomexit
python robustness_report.py --cheat-on-open --strategy-report Reports/williamscagr --strategy WilliamsR --data Data/forex_data/usdmxn.csv --symbol usdmxn --robustness-tests mcrandomentry mcrandomexit
python robustness_report.py --cheat-on-open --strategy-report Reports/williamscagr --strategy WilliamsR --data Data/forex_data/usdnzd.csv --symbol usdnzd --robustness-tests mcrandomentry mcrandomexit
python robustness_report.py --cheat-on-open --strategy-report Reports/williamscagr --strategy WilliamsR --data Data/forex_data/usdsek.csv --symbol usdsek --robustness-tests mcrandomentry mcrandomexit

python robustness_report.py --cheat-on-open --strategy-report Reports/williamssharpe --strategy WilliamsR --data Data/forex_data/usdaud.csv --symbol usdaud --robustness-tests mcrandomentry mcrandomexit
python robustness_report.py --cheat-on-open --strategy-report Reports/williamssharpe --strategy WilliamsR --data Data/forex_data/usdcad.csv --symbol usdcad --robustness-tests mcrandomentry mcrandomexit
python robustness_report.py --cheat-on-open --strategy-report Reports/williamssharpe --strategy WilliamsR --data Data/forex_data/usdchf.csv --symbol usdchf --robustness-tests mcrandomentry mcrandomexit
python robustness_report.py --cheat-on-open --strategy-report Reports/williamssharpe --strategy WilliamsR --data Data/forex_data/usdeur.csv --symbol usdeur --robustness-tests mcrandomentry mcrandomexit
python robustness_report.py --cheat-on-open --strategy-report Reports/williamssharpe --strategy WilliamsR --data Data/forex_data/usdgbp.csv --symbol usdgbp --robustness-tests mcrandomentry mcrandomexit
python robustness_report.py --cheat-on-open --strategy-report Reports/williamssharpe --strategy WilliamsR --data Data/forex_data/usdhkd.csv --symbol usdhkd --robustness-tests mcrandomentry mcrandomexit
python robustness_report.py --cheat-on-open --strategy-report Reports/williamssharpe --strategy WilliamsR --data Data/forex_data/usdjpy.csv --symbol usdjpy --robustness-tests mcrandomentry mcrandomexit
python robustness_report.py --cheat-on-open --strategy-report Reports/williamssharpe --strategy WilliamsR --data Data/forex_data/usdmxn.csv --symbol usdmxn --robustness-tests mcrandomentry mcrandomexit
python robustness_report.py --cheat-on-open --strategy-report Reports/williamssharpe --strategy WilliamsR --data Data/forex_data/usdnzd.csv --symbol usdnzd --robustness-tests mcrandomentry mcrandomexit
python robustness_report.py --cheat-on-open --strategy-report Reports/williamssharpe --strategy WilliamsR --data Data/forex_data/usdsek.csv --symbol usdsek --robustness-tests mcrandomentry mcrandomexit