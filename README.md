# Quant_Fin_Analysis

Quantitative financial analysis code.

## Repo Map

### Reproductions

This folder includes a list of jupyter notebooks. Each notebook includes a link to the source of the strategy and the original backtesting results from that source. It is followed by my recreation of that strategy and 3 of my backtests on that strategy. The first backtest has no Cheat On Open or Cheat On Close. This is the most realistic backtest. The second includes Cheat On Open. This is slightly less realistic as it gaurantees you get to buy on the next open price which is unlikely. The last includes Cheat On Close. This is the most unrealistic backtest because you can't possibly buy on a days close price. However, in some cases that is the backtest that lines up the most with the original source backtest so I include it.

**Disclaimer: I did not pay for anything on the source strategy websites. As a result not all of the details about the strategies are divulged, which results in some discrepancies between my backtest and the sources.**

### Data

### Graphs

### Reports

### Tests
Test files for various functions. Many functions rely heavily on backtrader and are therfore not tested by me because I couldn't possibly build test cases without using backtrader which would... well defeat the purpose of the test.

Most of the tests were assisted by GPT-4o.

### Main Files
