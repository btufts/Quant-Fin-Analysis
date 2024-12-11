import backtrader as bt
import math


class InMarketAnalyzer(bt.Analyzer):
    def __init__(self):
        self.total_bars = 0
        self.in_market_bars = 0
        self.total_percent_gain = 0.0
        self.total_gain = 0.0
        self.total_loss = 0.0
        self.cash = self.strategy.broker.get_cash()

    def next(self):
        self.total_bars += 1
        if self.strategy.position:  # If the strategy is in the market
            self.in_market_bars += 1

    def notify_trade(self, trade):
        if trade.isclosed:
            percent_gain = (trade.pnl / self.cash) * 100
            self.total_percent_gain += percent_gain
            if trade.pnl > 0:
                self.total_gain += trade.pnl
            else:
                self.total_loss += -trade.pnl

    def get_analysis(self):
        return {
            "Total Bars": self.total_bars,
            "Total In-Market Bars": self.in_market_bars,
            "Total Percent Gain": self.total_percent_gain,
            "Total Gains": self.total_gain,
            "Total Losses": self.total_loss,
        }


class CashValueAnalyzer(bt.Analyzer):
    def __init__(self):
        self.cash_values = []

    def next(self):
        self.cash_values.append(self.strategy.broker.getvalue())

    def get_analysis(self):
        return self.cash_values


class SortinoRatio(bt.Analyzer):
    def __init__(self):
        self.returns = []
        self.negative_returns = []
        self.cumulative_return = 0.0
        self.annual_risk_free_rate = 0  # Example risk-free rate (3% annually)
        self.hourly_risk_free_rate = (1 + self.annual_risk_free_rate) ** (
            1 / 8760
        ) - 1
        self.prev_value = self.strategy.broker.getvalue()

    def next(self):
        current_value = self.strategy.broker.getvalue()
        prev_value = self.prev_value
        self.prev_value = current_value
        hourly_return = (current_value - prev_value) / prev_value

        self.returns.append(hourly_return)

        # Track negative returns for downside risk calculation
        if hourly_return < self.hourly_risk_free_rate:
            self.negative_returns.append(
                hourly_return - self.hourly_risk_free_rate
            )

    def get_analysis(self):
        if not self.returns or not self.negative_returns:
            return {"Sortino Ratio": None}

        avg_return = sum(self.returns) / len(self.returns)
        downside_deviation = math.sqrt(
            sum(r**2 for r in self.negative_returns)
            / len(self.negative_returns)
        )

        # Calculate the Sortino ratio
        sortino_ratio = (
            (avg_return - self.hourly_risk_free_rate) / downside_deviation
            if downside_deviation != 0
            else None
        )

        return {"Sortino Ratio": sortino_ratio}
