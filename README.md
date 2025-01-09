# Quant_Fin_Analysis

Quantitative financial analysis code. Analysis of 4 strategies optimized for 10 currency pairs and robustness tested using 3 different methods. Many of the docstrings were assisted by Codeium.

## Setup

Developed with python version 3.12.2.

    pip install -r requirements.txt

## Data

Acquiring data can be done using download.py.

    python download.py --ticker AAPL --start_date 1995-01-01 --interval 1d --output_path Data/aapl_1d.csv

This fetches daily price data for Apple from 1995-01-01 to now and saves it in Data/aapl_1d.csv. This uses yfinance which uses Yahoo Finance! API to retrieve the data. This also supports intra-day data:

    python download.py --ticker AAPL --start_date 1995-01-01 --interval 1h --output_path Data/aapl_1h.csv

However, Yahoo Finance! will only provide intraday data for the last 730 days. The program will take care of fixing the start data if it is farther than that. To assist with compiling  intraday-data, download has another parameter "existing_file". This will append any new data to an existing csv and store it back in the same csv unless a new "output_path" is provided.

    python download.py --ticker AAPL --start_date 1995-01-01 --interval 1h --existing_file Data/aapl_1h.csv

All tickers come directly from Yahoo Finance!. See download.py for other optional parameters.

## Strategies

Strategies are defined in the Strategy module. There are a few default strategies included in the default_strategies submodule. When defining your own custom strategies, you can include them in the default_strategies submodule and they will automatically accessible by the rest of the program. You can also define a new submodule in which case you will need to import it into the __init__ of the Strategy Module.

### StrategyBase

New strategies must inherit from the StrategyBase class in the default_strategies submodule. There are two methods that must be implemented: long_condition(), close_condition(). Currently these are the only two positions that are supported. Short positions are in the pipeline. See the StrategyBase class for details on these two functions as well as other functions that can be overloaded.

Note: The StrategyBase class also comes with a function called get_optimization_args(). The default for this function just returns two empty dictionaries. In order to optimize a strategy you must fill in these dictionaries with some optimization args. At the moment this is the easiest way to do this. An improvement is in the pipeline.

## Main

main.py is the entry file to optimizing and robustness testing strategies. It has a single cmd line paramter config.

    python main.py --config Reports/example/config.yaml

See the example config to understand what types of experiments you can run and the paramters to pass. Each experiment is designed to evaluate strategies on a single security. Currently on a single strategy per experiment is supported but an increase is in the pipeline. The results of the experiment are stored in the same directory of as the config.yaml which is why it is encouraged to structure your directory in the same way as shown with Reports/example.
