from sqlite3 import paramstyle

from matplotlib import style
import backtrader as bt
import pandas as pd
import datetime

bt.observers.Trades
class FeedData(bt.feeds.GenericCSVData):
    lines=("chg","turnover",)
    params = (
        ("fromdate",datetime.datetime(2022,2,19)),
        ("todate",datetime.datetime(2022,7,29)),
        ('nullvalue', float('NaN')),
        ('dtformat', '%Y-%m-%d'),
        # ('tmformat', '%H:%M:%S'),
        ('datetime', 3),
        ('name', 0),
        ('code', 1),
        ('time', -1),
        ('open', 4),
        ("close",5),
        ('high', 6),
        ('low', 7),
        ('volume', 8),
        ('turnover', 9),
        ('amplitude', 10),
        ('chg', 11),
        ('turnover', 13),
        ('openinterest', -1),
    )


class TestStrategy(bt.Strategy):
 
    def log(self, txt, dt=None):
        ''' 提供记录功能'''
        dt = dt or self.datas[0].datetime.date(0)
        print('%s, %s' % (dt.isoformat(), txt))
 
    def __init__(self):
        # 引用到输入数据的close价格
        self.dataclose = self.datas[0].close
        self.dataopen = self.datas[0].open
        # self.order = self.order_target_percent(target=0.95)
        self.order = None
    def next(self):
        # 目前的策略就是简单显示下收盘价。
        print(self.dataclose[-1])
        if self.order:
            return
        # 检查是否在市场
        if not self.position:
            # 不在，那么连续3天价格下跌就买点
            if self.dataclose[-1] < self.dataclose[0]:
                    # 当前价格比上一次低
 
                    if self.dataclose[-2] < self.dataclose[-1]:
                        # 上一次的价格比上上次低
 
                        # 买入!!! 
                        # self.log('BUY CREATE, %.2f' % self.dataclose[0])
 
                        # Keep track of the created order to avoid a 2nd order
                        # self.order = self.order_target_percent(target=0.95)

                        self.order = self.buy(exectype=bt.Order.Market)
 
        else:
 
            # 已经在市场，5天后就卖掉。
            if len(self) >= (self.bar_executed ):#这里注意，Len(self)返回的是当前执行的bar数量，每次next会加1.而Self.bar_executed记录的最后一次交易执行时的bar位置。
                # SELL, SELL, SELL!!! (with all possible default parameters)
                # self.log('SELL CREATE, %.2f' % self.dataclose[0])
 
                # Keep track of the created order to avoid a 2nd order
                # self.order = self.order_target_percent(target=0.95)
                
                self.order = self.close(exectype=bt.Order.Market)

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
 
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log('Order Canceled/Margin/Rejected')
 
        # Write down: no pending order
        self.order = None

if __name__ == '__main__':
    cerebro = bt.Cerebro()

    cerebro.addstrategy(TestStrategy)

    #获取数据
    data = FeedData(dataname="info.csv",encoding="utf-8") # 加载数据
    cerebro.adddata(data,name="1.csv")  # 将数据传入回测系统
    
    cerebro.broker.setcash(100000.0)
    cerebro.broker.setcommission(commission=0.00015)
    cerebro.addsizer(bt.sizers.FixedSize, stake=1000)
    print('Starting Portfolio Value: %.2f' % cerebro.broker.getvalue())
    
    cerebro.addobserver(bt.observers.DrawDown)
    # cerebro.addobserver(bt.observers.Benchmark, data=data)
    # cerebro.addobserver(bt.observers.Broker)

    cerebro.run()
    cerebro.plot(
        style="candel",
        plotdist=0.1,
        barup = '#ff9896', bardown='#98df8a',
        volup='#ff9896', voldown='#98df8a',
        grid=False
    )
    print('Final Portfolio Value: %.2f' % cerebro.broker.getvalue())