import backtrader as bt
from backtrader.indicators import Indicator

# Copied from Backtrader for modification because it is not the same as strategy
# I am copying
class WilliamsR(Indicator):
    '''
    Developed by Larry Williams to show the relation of closing prices to
    the highest-lowest range of a given period.

    Known as Williams %R (but % is not allowed in Python identifiers)

    Formula:
      - num = highest_period - close
      - den = highestg_period - lowest_period
      - percR = (num / den) * -100.0

    See:
      - http://en.wikipedia.org/wiki/Williams_%25R
    '''
    lines = ('percR',)
    params = (('period', 2),
              ('upperband', -20.0),
              ('lowerband', -80.0),)

    plotinfo = dict(plotname='Williams R%')
    plotlines = dict(percR=dict(_name='R%'))

    def _plotinif(self):
        self.plotinfo.plotyhlines = [self.p.upperband, self.p.lowerband]

    def __init__(self):
        h = bt.indicators.Highest(self.data.high, period=self.p.period)
        l = bt.indicators.Lowest(self.data.low, period=self.p.period)
        c = self.data.close

        self.lines.percR = -100.0 * (h - c) / (h - l + 1e-8)

        super(WilliamsR, self).__init__()

# Some of this class was assisted using codeium autocomplete
class WilliamsRStrategy(bt.Strategy):
    params = (('period', 2),
              ('upperband', -20.0),
              ('lowerband', -80.0),
              ('cheat_on_open', False),)

    def log(self, txt, dt=None):
        ''' Logging function for this strategy'''
        dt = dt or self.datas[0].datetime.date(0)
        # print('%s, %s' % (dt.isoformat(), txt))

    def __init__(self):
        # Initialize Williams %R indicator
        self.williams_r = WilliamsR(self.datas[0], period=self.p.period, lowerband=self.p.lowerband, upperband=self.p.upperband)
        self.equity_curve = []
        self.buy_and_hold = []
        self.dates = []

        self.dataclose = self.datas[0].close
        self.datahigh = self.datas[0].high

        self.in_market = 0
        self.total_bars = 0

        self.total_percent_gain = 0
        self.total_gain = 0
        self.total_loss = 0

        self.buy_flag = False

    def notify_trade(self, trade):
        if trade.isclosed:
            percent_gain = (trade.pnl / self.cash) * 100
            self.total_percent_gain += percent_gain
            if trade.pnl > 0:
                self.total_gain += trade.pnl
            else:
                self.total_loss += -trade.pnl

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            # Buy/Sell order submitted/accepted to/by broker - Nothing to do
            return

        # Check if an order has been completed
        # Attention: broker could reject order if not enough cash
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log('BUY EXECUTED, %.2f' % order.executed.price)
            elif order.issell():
                self.log('SELL EXECUTED, %.2f' % order.executed.price)

            self.bar_executed = len(self)
        elif order.status == order.Canceled:
            if order.isbuy():
                self.log(f'Buy Order Canceled')
            else:
                self.log(f'Sell Order Canceled')
        elif order.status == order.Margin:
            if order.isbuy():
                self.log(f'Buy Order Margin')
            else:
                self.log(f'Sell Order Margin')

        elif order.status == order.Rejected:
            if order.isbuy():
                self.log(f'Buy Order Rejected')
            else:
                self.log(f'Sell Order Rejected')

        # Write down: no pending order
        self.order = None

    
    def next(self):
        self.equity_curve.append(self.broker.getvalue())
        self.dates.append(self.datas[0].datetime[0])

        self.total_bars += 1

        if not self.p.cheat_on_open:
            if not self.position:  # Not in the market
                if self.williams_r <= self.williams_r.p.lowerband:
                    self.cash = self.broker.getcash()
                    next_day_open = self.data.close[0]
                    size_to_buy = int(self.cash / next_day_open)
                    self.buy(exectype=bt.Order.Market, size=size_to_buy)

        if self.position:
            self.in_market += 1
            if self.williams_r.lines.percR[0] >= self.williams_r.p.upperband or self.datas[0].close[0] >= self.datas[0].high[-1]:
                self.close()
    
    def next_open(self):
        if not self.position:  # Not in the market
            if self.williams_r.lines.percR[0] <= self.williams_r.p.lowerband:
                self.cash = self.broker.getcash()
                next_day_open = self.data.open[0]
                size_to_buy = int(self.cash / next_day_open)
                self.buy(exectype=bt.Order.Market, size=size_to_buy)
