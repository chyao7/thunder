from re import S
from sqlite3 import paramstyle
# %matplotlib inline
# %matplotlib widget
# %matplotlib notebook
from matplotlib import style
import backtrader as bt
import pandas as pd
import datetime
import os

from backtrader.observers import benchmark
class BasicData(bt.feeds.GenericCSVData):
    params = (
        ("fromdate",datetime.datetime(2022,1,1)),
        ("todate",datetime.datetime(2023,8,30)),
        ('nullvalue', float('0')),
        ('dtformat', '%Y-%m-%d'),
        # ('tmformat', '%H:%M:%S'),
        ('datetime', 3),
        ('name', 1),
        ('code', 2),
        ('time', -1),
        ('open', 4),
        ("close",5),
        ('high', 6),
        ('low', 7),
        ('volume', 8),
        ('openinterest', -1),
    )


class FeedData(bt.feeds.GenericCSVData):
    lines=("chg","turnover",)
    params = (
        ("fromdate",datetime.datetime(2022,1,1)),
        ("todate",datetime.datetime(2023,8,30)),
        ('nullvalue', float('0')),
        ('dtformat', '%Y-%m-%d'),
        # ('tmformat', '%H:%M:%S'),
        ('datetime', 1),

        ('name', 2),
        ('code', 3),
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
        dt = dt or self.datas[-1].datetime.date(0)
        print('%s, %s' % (dt.isoformat(), txt))
 
    def __init__(self):
        self.res = pd.read_csv("./result.csv")
        # self.res["date"]= pd.to_datetime(self.res["date"])
        self.ind = {}
        self.order = None
        self.acc = 0
        self.all = 0
        
        # for i,d in enumerate(self.datas):
            # self.ind[d]=bt.indicators.MACD(d)
    def getps(self,p,index):
        # 分配权重
        if index==0:
            money = self.broker.getvalue()*0.7
        else:

            money = self.broker.getvalue()*0.3
        return int((money/p)/100)*100    

    def next(self):
        print("\n\n","*"*20,len(self),"*"*20)
        if self.all!=0:
            print(self.acc/self.all,self.all,self.acc,self.all)
        # print(self.res.head(-5))

        d_info = str(self.datas[0].datetime.date(0))
        info = self.res[self.res["date"]==d_info]
        buy_list = ["%06d" %i for i in info["code"]]
        self.score_list = [float(i) for i in info["res"]]
        print(buy_list)
        # print(self._trades(d).barlen())

        for d in self.datas:
            if d.close[0]==0.0:
                continue
            if self.getposition(d).size:
                if d._name not in buy_list:
                    self.order = self.sell(data=d)
                    self.log(f'SEll {d._name}, %.2f' % d.close[0])
                # elif score_list[buy_list.index(d._name)] < 0.75:
                #     self.order = self.sell(data=d)
                #     self.log(f'SEll {d._name}, %.2f' % d.close[0])
        
        for i in buy_list:
            for d in self.datas:
                if i==d._name:
                    if d.close[0]==0.0 :
                        continue
            # if score_list[buy_list.index(d._name)]>=0.8:
                    self.log(f'BUY {d._name}, {d.close[0]}' )
                    self.order = self.buy(data=d,size=self.getps(d.close[0],buy_list.index(d._name)))
            
    def notify_trade(self, trade):
        if not trade.isclosed:
            return
        self.log(f'执行标的：{trade.getdataname()}，策略收益：毛收益 {trade.pnl:.2f}, 净收益 {trade.pnlcomm:.2f},')
        self.all+=1
        if trade.pnl>=0:
            self.acc+=1

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            # Buy/Sell order submitted/accepted to/by broker - Nothing to do
            return
        # Check if an order has been completed
        # Attention: broker could reject order if not enough cash
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(
                    'BUY EXECUTED, Price: %.2f, Cost: %.2f, Comm %.2f, Value %.2f' %
                    (order.executed.price,
                     order.executed.value,
                     order.executed.comm,
                     self.broker.getvalue()
                     ))
 
                self.buyprice = order.executed.price
                self.buycomm = order.executed.comm
            else:  # Sell
                self.log('SELL EXECUTED, Price: %.2f, Cost: %.2f, Comm %.2f, Leftcash%.2f' %
                         (order.executed.price,
                          order.executed.value,
                          order.executed.comm,
                          self.broker.getcash()))
 
            self.bar_executed = len(self)
 
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log('Order Canceled/Margin/Rejected')
 
        self.order = None


def backtest():
    # import matplotlib.pyplot as plt
    # plt.rcParams['figure.figsize']=(22.8, 14.2)
    codepool = "沪深300"
    # codepool = "中证200"

    root = "./data"
    cerebro = bt.Cerebro(stdstats=False)
    cerebro.addstrategy(TestStrategy)
    #获取数据
    files = os.listdir(root)
    from efinance.stock import get_members
    data = get_members(codepool)["股票代码"]
    # print(data)
     # 将数据传入回测系�?
    
    for file in data:
        file = file+".csv"
#         print(file)
        path = os.path.join(root,file)
        if not os.path.exists(path):
            continue
        data = FeedData(dataname=path,encoding="utf-8") # 加载数据
        cerebro.adddata(data,name=file[:-4])  # 将数据传入回测系�?
    cerebro.broker.set_coc(True)
    cerebro.broker.setcash(100000.0)
    cerebro.broker.setcommission(commission=0.00015)
    cerebro.addsizer(bt.sizers.PercentSizerInt, percents=45)
    # cerebro.addsizer(bt.sizers.FixedSize, stake=500)
    
   
    # 查看收益序列
    cerebro.addobserver(bt.observers.TimeReturn)
    # 查看回撤序列
    cerebro.addobserver(bt.observers.DrawDown)
    # backtrader.observers.Broker
    cerebro.addobserver(bt.observers.Value)
    cerebro.addobserver(bt.observers.Trades)
    cerebro.addobserver(bt.observers.Broker)
    # 添加banchmark
    benchdata = BasicData(dataname="./data/上证指数.csv",encoding="utf-8")
    cerebro.adddata(benchdata,name="spy")
    cerebro.addobserver(bt.observers.Benchmark,data=benchdata)
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='SharpeRatio')
    print('Starting Portfolio Value: %.2f' % cerebro.broker.getvalue())
    # base = bt.feeds.PandasData(dataname=data, fromdate=st_date, todate=ed_date)
    cerebro.run()
    for d in cerebro.datas[:-1]:
        d.plotinfo.plot = False
    
    print('Final Portfolio Value: %.2f' % cerebro.broker.getvalue())
    # cerebro.plot(
    #     style="candel",
    #     plotdist=0.1,
    #     barup = 'r', bardown='g',
    #     volup='r', voldown='g',
    #     grid=True,
    # )

if __name__ == '__main__':
    backtest()

    # 132251.55
