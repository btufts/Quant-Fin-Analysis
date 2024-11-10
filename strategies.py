# system imports
import random
from abc import abstractmethod

# Third party imports
import backtrader as bt
from indicators import WilliamsR

class StrategyBase(bt.Strategy):
    @staticmethod
    def get_optimization_args():
        '''
        This function will take key word parameters and return lists for optimization
            This allows for dynamic use of strategy_report.py
        This will need to be defined for every strategy that you want to build a
            strategy report for. Build it how you want to use it.
            See strategy_report.py opt_universe() to see how this is used
        
        Returns:
            dict(str, list): A dictionary of parameters to be optimized and their values
            dict(str, int): A dictionary of parameters to be optimized and their step sizes
        '''
        return {}, {}
    
    @abstractmethod
    def long_condition(self):
        '''
        The purpose of this is to put the condition into a function that can be overwritten for robustness testing
        '''
        pass

    @abstractmethod
    def close_condition(self):
        '''
        The purpose of this is to put the condition into a function that can be overwritten for robustness testing
        '''
        pass

# Some of this class was assisted using codeium autocomplete
class WilliamsRStrategy(StrategyBase):
    params = (('period', 2),
              ('upperband', -20.0),
              ('lowerband', -80.0),)

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

        self.cheating = self.cerebro.p.cheat_on_open

    @staticmethod
    def get_optimization_args(**kwargs) -> tuple[dict[str, list], dict[str, int]]:
        opt_args = {}
        if "min_period" in kwargs or "max_period" in kwargs:
            if "max_period" not in kwargs:
                exit("max_period must be specified if min_period is specified")
            if "min_period" not in kwargs:
                exit("min_period must be specified if max_period is specified")
            opt_args['period']=list(range(
                int(kwargs['min_period']),
                int(kwargs['max_period']),
                int(kwargs['period_step']) if 'period_step' in kwargs else 1
            ))
        elif "period" in kwargs:
            opt_args['period']=[kwargs['period']]
        
        if "min_lowerband" in kwargs or "max_lowerband" in kwargs:
            if "max_lowerband" not in kwargs:
                exit("max_lowerband must be specified if min_lowerband is specified")
            if "min_lowerband" not in kwargs:
                exit("min_lowerband must be specified if max_lowerband is specified")
            opt_args['lowerband']=list(range(
                int(kwargs['min_lowerband']),
                int(kwargs['max_lowerband']),
                int(kwargs['lowerband_step']) if 'lowerband_step' in kwargs else -5
            ))
        elif "lowerband" in kwargs:
            opt_args['lowerband']=[kwargs['lowerband']]
        
        if "min_upperband" in kwargs or "max_upperband" in kwargs:
            if "max_upperband" not in kwargs:
                exit("max_upperband must be specified if min_upperband is specified")
            if "min_upperband" not in kwargs:
                exit("min_upperband must be specified if max_upperband is specified")
            opt_args['upperband']=list(range(
                int(kwargs['min_upperband']),
                int(kwargs['max_upperband']),
                int(kwargs['upperband_step']) if 'upperband_step' in kwargs else -5
            ))
        elif "upperband" in kwargs:
            opt_args['upperband']=[kwargs['upperband']]

        steps = {
            'period': int(kwargs['period_step']) if 'period_step' in kwargs else 1,
            'lowerband': int(kwargs['lowerband_step']) if 'lowerband_step' in kwargs else -5,
            'upperband': int(kwargs['upperband_step']) if 'upperband_step' in kwargs else -5
        }

        return opt_args, steps

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

    def long_condition(self):
        if self.williams_r <= self.williams_r.p.lowerband:
            return True
        else:
            return False
        
    def close_condition(self):
        if self.williams_r.lines.percR[0] >= self.williams_r.p.upperband or self.datas[0].close[0] >= self.datas[0].high[-1]:
            return True
        else:
            return False
    
    def next(self):
        self.equity_curve.append(self.broker.getvalue())
        self.dates.append(self.datas[0].datetime[0])

        self.total_bars += 1

        if not self.cheating:
            if not self.position and self.long_condition():  # Not in the market
                self.cash = self.broker.getcash()
                next_day_open = self.data.close[0]
                size_to_buy = int(self.cash / next_day_open)
                self.buy(exectype=bt.Order.Market, size=size_to_buy)

        if self.position:
            self.in_market += 1
            if self.close_condition():
                self.close()
    
    def next_open(self):
        if not self.position and self.long_condition():  # Not in the market
            self.cash = self.broker.getcash()
            next_day_open = self.data.open[0]
            size_to_buy = int(self.cash / next_day_open)
            self.buy(exectype=bt.Order.Market, size=size_to_buy)

class RandomStrategy(StrategyBase):
    def __init__(self):
        self.order = None
        self.cheating = self.cerebro.p.cheat_on_open
        self.action = None

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return False
        self.order = None
        return True
    
    def long_condition(self):
        if self.action:
            return True
        else:
            return False
    
    def close_condition(self):
        if self.action:
            return True
        else:
            return False
        
    def next(self):
        self.action = random.randint(0,2) # Get random value 0: hold, 1: change position
        if self.order:
            return  # Skip if there's a pending order
    
        if not self.cheating:
            if self.long_condition():
                if not self.position:  # Not in the market
                    self.buy(exectype=bt.Order.Market)
                else:
                    self.close()

    def next_open(self):
        if self.long_condition():
            if not self.position:  # Not in the market
                self.buy(exectype=bt.Order.Market)
            else:
                self.close()