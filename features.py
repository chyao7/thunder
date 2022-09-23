import multitasking
from retry import retry
from efinance.stock import get_members
import os
import pandas as pd
import talib as ta
import numpy as np

def calculate():
    @multitasking.task
    @retry(tries=3, delay=1)
    def process(d):
        d.rename(columns = {"日期": "date","股票名称":"name","股票代码":"code","开盘":"open","收盘":"close","最高":"high","最低":"low","成交量":"volume"},  inplace=True)
        d['date'] = pd.to_datetime(d['date'])
        d = d.set_index('date')
        d = d['2016-01-01':'2022-08-18']
        # d = d['2017-01-01':'2022-01-01']


        d = d.dropna()
        if len(d)<100:
            return
   
        # 移动均线
        d["ema_30"] = ta.EMA(d.close, timeperiod=30)
        d["ema_10"] = ta.EMA(d.close, timeperiod=10)
        d["ema_5"] = ta.EMA(d.close, timeperiod=5)
        d["adrx_14"] = ta.ADXR(d.high, d.low, d.close, timeperiod=14)
        d["adrx_5"] = ta.ADXR(d.high, d.low, d.close, timeperiod=5)
        d["apo"] = ta.APO(d.close, fastperiod=5, slowperiod=20, matype=0)
        d["CCI_14"] = ta.CCI(d.high, d.low, d.close, timeperiod=14)
        d["CCI_4"] = ta.CCI(d.high, d.low, d.close, timeperiod=4)
        d["MFI_10"] = ta.MFI(d.high, d.low, d.close, d.volume, timeperiod=10)
        d["MFI_3"] = ta.MFI(d.high, d.low, d.close, d.volume, timeperiod=3)
        d["PLUS_DI_10"] = ta.PLUS_DI(d.high, d.low, d.close, timeperiod=10)
        d["PLUS_DI_3"] = ta.PLUS_DI(d.high, d.low, d.close, timeperiod=3)
        d["RSI_14"]=ta.RSI(d.close, timeperiod=14)
        d["RSI_5"]=ta.RSI(d.close, timeperiod=5)
        d["RSI_3"]=ta.RSI(d.close, timeperiod=3)
        d["ROC_7"] = ta.ROC(d.close, timeperiod=7)
        d["ROC_3"] = ta.ROC(d.close, timeperiod=3)
        d["ADOSC_3"] = ta.ADOSC(d.high, d.low, d.close, d.volume, fastperiod=3, slowperiod=10)
        d["ADOSC_10"] = ta.ADOSC(d.high, d.low, d.close, d.volume, fastperiod=10, slowperiod=30)
        d["HT_DCPERIOD"] = ta.HT_DCPERIOD(d.close)
        d["NATR_14"] = ta.NATR(d.high, d.low, d.close, timeperiod=14)
        d["NATR_7"] = ta.NATR(d.high, d.low, d.close, timeperiod=7)
        d["NATR_3"] = ta.NATR(d.high, d.low, d.close, timeperiod=3)
        a = ( d.close.shift(-1)-d.close)/ d.close
        b = d.code    
        d = d.drop(columns=["code",'Unnamed: 0',"name"])
        d = d.apply(lambda x: (x - np.mean(x)) / (np.std(x)))
        d["y"] = a*100
        d["code"] = b
        d_train = d['2016-01-01':'2022-01-01']
        d_train = d_train.dropna(axis=0)
        d_train = d_train.drop(columns=["code"])

        d_test = d['2022-01-01':'2022-8-20']
        d_test = d_test.drop(columns=["y"])
        d_test = d_test.dropna(axis=0)
        print(d.columns)

        all_train.append(d_train)
        all_test.append(d_test)
        print("X"*100)
        print("train",d_train.head(5))
        print("tets",d_test.head(5))
        return
    all_train = []
    all_test = []
    datas = get_members("中证200")["股票代码"]
    for code in datas:
        path = os.path.join("./data",code+".csv")
        if not os.path.exists(path):
            print(path)
            continue
        df = pd.read_csv(path)
        process(df)
    multitasking.wait_for_tasks()
    features_train = pd.concat(all_train, axis=0)
    features_test = pd.concat(all_test, axis=0)
    features_train.to_csv("data.csv", index=None)
    features_test.to_csv("data2.csv")



if __name__=="__main__":
    calculate()


