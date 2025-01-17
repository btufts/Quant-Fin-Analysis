# system imports
import random
from abc import abstractmethod

# Third party imports
import backtrader as bt
from indicators import WilliamsR


class StrategyBase(bt.Strategy):

    def __init__(self):
        self.cheating = self.cerebro.p.cheat_on_open
        self.equity_curve = []
        self.dates = []
        self.datetimes = []

        # Indicators to be used with cheat_on_open
        # next() sets them, next_open() uses them
        self.coo_long_indicator = False
        self.coo_close_indicator = False

    @staticmethod
    def get_optimization_args() -> tuple[dict[str, list], dict[str, int]]:
        """
        This function will take key word parameters and
        return lists for optimization
            This allows for dynamic use of strategy_report.py
        This will need to be defined for every strategy that you want to build
        a strategy report for. Build it how you want to use it.
            See strategy_report.py opt_universe() to see how this is used

        Returns:
            dict(str, list): A dictionary of parameters to be optimized and
            their values
            dict(str, int): A dictionary of parameters to be optimized and
            their step sizes
        """
        return {}, {}

    def log(self, txt, dt=None):
        """Logging function for this strategy"""
        dt = dt or self.datas[0].datetime.date(0)
        # print("%s, %s" % (dt.isoformat(), txt))

    # def notify_trade(self, trade):
    #     if trade.isclosed:
    #         percent_gain = (trade.pnl / self.cash) * 100
    #         self.total_percent_gain += percent_gain
    #         if trade.pnl > 0:
    #             self.total_gain += trade.pnl
    #         else:
    #             self.total_loss += -trade.pnl

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            # Buy/Sell order submitted/accepted to/by broker - Nothing to do
            return

        # Check if an order has been completed
        # Attention: broker could reject order if not enough cash
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log("BUY EXECUTED, %f" % order.executed.price)
            elif order.issell():
                self.log("SELL EXECUTED, %f" % order.executed.price)

            self.bar_executed = len(self)
        elif order.status == order.Canceled:
            if order.isbuy():
                self.log("Buy Order Canceled")
            else:
                self.log("Sell Order Canceled")
        elif order.status == order.Margin:
            if order.isbuy():
                self.log("Buy Order Margin")
            else:
                self.log("Sell Order Margin")

        elif order.status == order.Rejected:
            if order.isbuy():
                self.log("Buy Order Rejected")
            else:
                self.log("Sell Order Rejected")

        # Write down: no pending order
        self.order = None

    @abstractmethod
    def long_condition(self):
        """
        The purpose of this is to put the condition into a function
        that can be overwritten for robustness testing
        """
        pass

    @abstractmethod
    def close_condition(self):
        """
        The purpose of this is to put the condition into a function that can
        be overwritten for robustness testing
        """
        pass

    def next(self):
        self.equity_curve.append(self.broker.getvalue())
        self.dates.append(self.data.datetime.date(0))
        self.datetimes.append(self.data.datetime.datetime(0))
        if not self.cheating:
            if not self.position and self.long_condition():
                self.buy()
            elif self.position and self.close_condition():
                self.close()
        else:
            if not self.position and self.long_condition():
                self.coo_long_indicator = True
            elif self.position and self.close_condition():
                self.coo_close_indicator = True

    def next_open(self):
        if self.coo_long_indicator:
            self.coo_long_indicator = False
            self.cash = self.broker.getcash()
            next_day_open = self.data.open[0]
            size_to_buy = self.cash / next_day_open
            self.buy(size=size_to_buy)
        if self.coo_close_indicator:
            self.coo_close_indicator = False
            self.close()


# Some of this class was assisted using codeium autocomplete
class WilliamsRStrategy(StrategyBase):
    params = (
        ("period", 2),
        ("upperband", -20.0),
        ("lowerband", -80.0),
    )

    def __init__(self):

        super().__init__()
        # Initialize Williams %R indicator
        self.williams_r = WilliamsR(
            self.datas[0],
            period=self.p.period,
            lowerband=self.p.lowerband,
            upperband=self.p.upperband,
        )
        self.buy_and_hold = []

        self.dataclose = self.datas[0].close
        self.datahigh = self.datas[0].high

        self.in_market = 0
        self.total_bars = 0
        self.total_percent_gain = 0
        self.total_gain = 0
        self.total_loss = 0

    @staticmethod
    def get_optimization_args(
        **kwargs,
    ) -> tuple[dict[str, list], dict[str, int]]:
        opt_args = {}

        steps = {
            "period": (
                int(kwargs["period_step"]) if "period_step" in kwargs else 1
            ),
            "lowerband": (
                int(kwargs["lowerband_step"])
                if "lowerband_step" in kwargs
                else -5
            ),
            "upperband": (
                int(kwargs["upperband_step"])
                if "upperband_step" in kwargs
                else -5
            ),
        }

        if "min_period" in kwargs or "max_period" in kwargs:
            if "max_period" not in kwargs:
                raise ValueError(
                    "max_period must be specified if min_period is specified"
                )
            if "min_period" not in kwargs:
                raise ValueError(
                    "min_period must be specified if max_period is specified"
                )
            opt_args["period"] = list(
                range(
                    int(kwargs["min_period"]),
                    int(kwargs["max_period"]) + steps["period"],
                    (steps["period"]),
                )
            )
        elif "period" in kwargs:
            opt_args["period"] = [kwargs["period"]]

        if "min_lowerband" in kwargs or "max_lowerband" in kwargs:
            if "max_lowerband" not in kwargs:
                raise ValueError(
                    "max_lowerband must be specified if "
                    + "min_lowerband is specified"
                )
            if "min_lowerband" not in kwargs:
                raise ValueError(
                    "min_lowerband must be specified if "
                    + "max_lowerband is specified"
                )
            opt_args["lowerband"] = list(
                range(
                    int(kwargs["min_lowerband"]),
                    int(kwargs["max_lowerband"]) + steps["lowerband"],
                    (steps["lowerband"]),
                )
            )
        elif "lowerband" in kwargs:
            opt_args["lowerband"] = [kwargs["lowerband"]]

        if "min_upperband" in kwargs or "max_upperband" in kwargs:
            if "max_upperband" not in kwargs:
                raise ValueError(
                    "max_upperband must be specified if"
                    + " min_upperband is specified"
                )
            if "min_upperband" not in kwargs:
                raise ValueError(
                    "min_upperband must be specified if "
                    + "max_upperband is specified"
                )
            opt_args["upperband"] = list(
                range(
                    int(kwargs["min_upperband"]),
                    int(kwargs["max_upperband"]) + steps["upperband"],
                    (steps["upperband"]),
                )
            )
        elif "upperband" in kwargs:
            opt_args["upperband"] = [kwargs["upperband"]]

        return opt_args, steps

    def long_condition(self):
        if self.williams_r < self.williams_r.p.lowerband:
            return True
        else:
            return False

    def close_condition(self):
        if (
            self.williams_r.lines.percR[0] > self.williams_r.p.upperband
            or self.datas[0].close[0] > self.datas[0].high[-1]
        ):
            return True
        else:
            return False


class RandomStrategy(StrategyBase):
    def __init__(self):
        super().__init__()
        self.order = None
        self.cheating = self.cerebro.p.cheat_on_open
        self.action = None

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return False
        self.order = None
        return True

    def long_condition(self):
        if self.action == 2:
            return True
        else:
            return False

    def close_condition(self):
        if self.action == 1:
            return True
        else:
            return False

    def next(self):
        self.equity_curve.append(self.broker.getvalue())
        self.dates.append(self.data.datetime.date(0))
        self.datetimes.append(self.data.datetime.datetime(0))
        self.action = random.randint(
            0, 3
        )  # Get random value 0: hold, 1: change position
        if self.order:
            return  # Skip if there's a pending order

        if not self.cheating:
            if not self.position and self.long_condition():
                self.buy(exectype=bt.Order.Market)
            if self.position and self.close_condition():
                self.close()
        else:
            if not self.position and self.long_condition():
                self.coo_long_indicator = True
            if self.position and self.close_condition():
                self.coo_close_indicator = True


class CCIStrategy(StrategyBase):
    params = (
        ("period", 20),  # CCI period for calculation
        ("lowerband", -100),  # Oversold threshold
    )

    def __init__(self):
        # Initialize the CCI indicator
        super().__init__()
        self.cci = bt.indicators.CommodityChannelIndex(
            period=self.params.period
        )

    @staticmethod
    def get_optimization_args(
        **kwargs,
    ) -> tuple[dict[str, list], dict[str, int]]:
        opt_args = {}
        if "min_period" in kwargs or "max_period" in kwargs:
            if "max_period" not in kwargs:
                raise ValueError(
                    "max_period must be specified if min_period is specified"
                )
            if "min_period" not in kwargs:
                raise ValueError(
                    "min_period must be specified if max_period is specified"
                )
            opt_args["period"] = list(
                range(
                    int(kwargs["min_period"]),
                    int(kwargs["max_period"]),
                    (
                        int(kwargs["period_step"])
                        if "period_step" in kwargs
                        else 1
                    ),
                )
            )
        elif "period" in kwargs:
            opt_args["period"] = [kwargs["period"]]

        if "min_lowerband" in kwargs or "max_lowerband" in kwargs:
            if "max_lowerband" not in kwargs:
                raise ValueError(
                    "max_lowerband must be specified "
                    + "if min_lowerband is specified"
                )
            if "min_lowerband" not in kwargs:
                raise ValueError(
                    "min_lowerband must be specified "
                    + "if max_lowerband is specified"
                )
            opt_args["lowerband"] = list(
                range(
                    int(kwargs["min_lowerband"]),
                    int(kwargs["max_lowerband"]),
                    (
                        int(kwargs["lowerband_step"])
                        if "lowerband_step" in kwargs
                        else -5
                    ),
                )
            )
        elif "lowerband" in kwargs:
            opt_args["lowerband"] = [kwargs["lowerband"]]

        steps = {
            "period": (
                int(kwargs["period_step"]) if "period_step" in kwargs else 1
            ),
            "lowerband": (
                int(kwargs["lowerband_step"])
                if "lowerband_step" in kwargs
                else -5
            ),
        }

        return opt_args, steps

    def long_condition(self):
        if self.cci[0] < self.params.lowerband:
            return True
        else:
            return False

    def close_condition(self):
        if self.data.close[0] > self.data.high[-1]:
            return True
        else:
            return False


class StochasticStrategy(StrategyBase):
    params = (
        ("period", 14),
        ("period_dfast", 3),
        ("lowerband", 20),
    )

    def __init__(self):
        super().__init__()
        self.stochastic = bt.indicators.StochasticFast(
            self.data,
            period=self.p.period,
            period_dfast=self.p.period_dfast,
            safediv=True,
        )

    @staticmethod
    def get_optimization_args(
        **kwargs,
    ) -> tuple[dict[str, list], dict[str, int]]:
        opt_args = {}
        if "min_period" in kwargs or "max_period" in kwargs:
            if "max_period" not in kwargs:
                raise ValueError(
                    "max_period must be specified if min_period is specified"
                )
            if "min_period" not in kwargs:
                raise ValueError(
                    "min_period must be specified if max_period is specified"
                )
            opt_args["period"] = list(
                range(
                    int(kwargs["min_period"]),
                    int(kwargs["max_period"]),
                    (
                        int(kwargs["period_step"])
                        if "period_step" in kwargs
                        else 1
                    ),
                )
            )
        elif "period" in kwargs:
            opt_args["period"] = [kwargs["period"]]

        if "min_period_dfast" in kwargs or "max_period_dfast" in kwargs:
            if "max_period" not in kwargs:
                raise ValueError(
                    "max_period_dfast must be specified if "
                    + "min_period_dfast is specified"
                )
            if "min_period" not in kwargs:
                raise ValueError(
                    "min_period_dfast must be specified if "
                    + "max_period_dfast is specified"
                )
            opt_args["period_dfast"] = list(
                range(
                    int(kwargs["min_period_dfast"]),
                    int(kwargs["max_period_dfast"]),
                    (
                        int(kwargs["period_dfast_step"])
                        if "period_dfast_step" in kwargs
                        else 1
                    ),
                )
            )
        elif "period_dfast" in kwargs:
            opt_args["period_dfast"] = [kwargs["period_dfast"]]

        if "min_lowerband" in kwargs or "max_lowerband" in kwargs:
            if "max_lowerband" not in kwargs:
                raise ValueError(
                    "max_lowerband must be specified "
                    + "if min_lowerband is specified"
                )
            if "min_lowerband" not in kwargs:
                raise ValueError(
                    "min_lowerband must be specified "
                    + "if max_lowerband is specified"
                )
            opt_args["lowerband"] = list(
                range(
                    int(kwargs["min_lowerband"]),
                    int(kwargs["max_lowerband"]),
                    (
                        int(kwargs["lowerband_step"])
                        if "lowerband_step" in kwargs
                        else -5
                    ),
                )
            )
        elif "lowerband" in kwargs:
            opt_args["lowerband"] = [kwargs["lowerband"]]

        steps = {
            "period": (
                int(kwargs["period_step"]) if "period_step" in kwargs else 1
            ),
            "period_dfast": (
                int(kwargs["period_dfast_step"])
                if "period_dfast_step" in kwargs
                else 1
            ),
            "lowerband": (
                int(kwargs["lowerband_step"])
                if "lowerband_step" in kwargs
                else -5
            ),
        }

        return opt_args, steps

    def long_condition(self):
        if self.stochastic.lines.percD[0] < self.p.lowerband:
            return True
        return False

    def close_condition(self):
        if self.datas[0].close[0] > self.datas[0].high[-1]:
            return True
        return False


class TurnaroundTuesday(StrategyBase):
    # number of days since entry to wait
    params = (("wait_days", 5),)

    def __init__(self):
        super().__init__()
        self.days_since_entry = 0

    @staticmethod
    def get_optimization_args(
        **kwargs,
    ) -> tuple[dict[str, list], dict[str, int]]:
        opt_args = {}
        if "min_wait_days" in kwargs or "max_wait_days" in kwargs:
            if "max_wait_days" not in kwargs:
                raise ValueError(
                    "max_wait_days must be specified if "
                    + "min_wait_days is specified"
                )
            if "min_wait_days" not in kwargs:
                raise ValueError(
                    "min_wait_days must be specified if "
                    + "max_wait_days is specified"
                )
            opt_args["wait_days"] = list(
                range(
                    int(kwargs["min_wait_days"]),
                    int(kwargs["max_wait_days"]),
                    (
                        int(kwargs["wait_days_step"])
                        if "wait_days_step" in kwargs
                        else 1
                    ),
                )
            )
        elif "wait_days" in kwargs:
            opt_args["wait_days"] = [kwargs["wait_days"]]

        steps = {
            "wait_days": (
                int(kwargs["wait_days_step"])
                if "wait_days_step" in kwargs
                else 1
            )
        }

        return opt_args, steps

    def long_condition(self):
        # this function is only called without a position
        # so reset days since entry
        if self.data.datetime.date().weekday() == 0:
            # Check if close is lower
            if self.datas[0].close[0] < self.datas[0].close[-1]:
                return True
        return False

    def close_condition(self):
        # Function is only called when we have position
        # so increment days since entry
        if (
            self.days_since_entry == self.p.wait_days
            or self.datas[0].close[0] > self.datas[0].high[-1]
        ):
            # Exit position after 5 trading days
            return True
        return False

    def next(self):
        self.equity_curve.append(self.broker.getvalue())
        self.dates.append(self.data.datetime.date(0))
        self.datetimes.append(self.data.datetime.datetime(0))

        if not self.cheating:
            if not self.position and self.long_condition():
                self.buy()
                self.days_since_entry = 0
            if self.position:
                self.days_since_entry += 1
                if self.close_condition():
                    self.close()
        else:
            if not self.position and self.long_condition():
                self.coo_long_indicator = True
                self.days_since_entry = 0
            if self.position:
                self.days_since_entry += 1
                if self.close_condition():
                    self.coo_close_indicator = True
