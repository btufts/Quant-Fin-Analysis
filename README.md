# Quant_Fin_Analysis

Quantitative financial analysis code. Analysis of 4 strategies optimized for 10 currency pairs and robustness tested using 3 different methods. Many of the docstrings were assisted by Codeium.

## Setup

Developed with python version 3.12.2.

    pip install -r requirements.txt

## Strategies

Strategies are defined in the Strategy module. There are a few default strategies included in the default_strategies submodule. When defining your own custom strategies, you can include them in the default_strategies submodule and they will automatically accessible by the rest of the program. You can also define a new submodule in which case you will need to import it into the __init__ of the Strategy Module.

### StrategyBase

New strategies must inherit from the StrategyBase class in the default_strategies submodule. There are two methods that must be implemented: long_condition(), close_condition(). Currently these are the only two positions that are supported. Short positions are in the pipeline. See the StrategyBase class for details on these two functions as well as other functions that can be overloaded.

Note: The StrategyBase class also comes with a function called get_optimization_args(). The default for this function just returns two empty dictionaries. In order to optimize a strategy you must fill in these dictionaries with some optimization args. At the moment this is the easiest way to do this. An improvement is in the pipeline.
