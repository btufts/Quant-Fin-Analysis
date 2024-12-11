# Third party imports
import backtrader as bt
from backtrader.indicators import Indicator

# Copied from Backtrader for modification
# because it is not the same as strategy
# I am copying


class WilliamsR(Indicator):
    """
    Developed by Larry Williams to show the relation of closing prices to
    the highest-lowest range of a given period.

    Known as Williams %R (but % is not allowed in Python identifiers)

    Formula:
      - num = highest_period - close
      - den = highestg_period - lowest_period
      - percR = (num / den) * -100.0

    See:
      - http://en.wikipedia.org/wiki/Williams_%25R
    """

    lines = ("percR",)
    params = (
        ("period", 2),
        ("upperband", -20.0),
        ("lowerband", -80.0),
    )

    plotinfo = dict(plotname="Williams R%")
    plotlines = dict(percR=dict(_name="R%"))

    def _plotinif(self):
        self.plotinfo.plotyhlines = [self.p.upperband, self.p.lowerband]

    def __init__(self):
        period_high = bt.indicators.Highest(
            self.data.high, period=self.p.period
        )
        period_low = bt.indicators.Lowest(self.data.low, period=self.p.period)
        c = self.data.close

        self.lines.percR = (
            -100.0 * (period_high - c) / (period_high - period_low + 1e-8)
        )

        super(WilliamsR, self).__init__()
