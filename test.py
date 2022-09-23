import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
import pandas as pd

import multitasking
from retry import retry
from efinance.stock import get_members
import os
import pandas as pd
import talib as ta
import numpy as np

def calculate(name,start,end,):
    @multitasking.task
    @retry(tries=3, delay=1)
    def process(d):
        d.rename(columns = {"日期": "date","股票名称":"name","股票代码":"code","开盘":"open","收盘":"close","最高":"high","最低":"low","成交量":"volume"},  inplace=True)
        d['date'] = pd.to_datetime(d['date'])
        d = d.set_index('date')
        d = d[start:end]
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
        a = d.code
        d = d.drop(columns=["code",'Unnamed: 0',"name"])
        d = d.apply(lambda x: (x - np.mean(x)) / (np.std(x)))
        d["code"] = a
        all.append(d)            
        return
    all = []
    datas = get_members(name)["股票代码"]
    for code in datas:
        path = os.path.join("./data",code+".csv")
        if not os.path.exists(path):
            print(path)
            continue
        df = pd.read_csv(path)
        process(df)
    multitasking.wait_for_tasks()
    features = pd.concat(all, axis=0)
    return features

def test():
    model_path = "./checkpoint/lgb2.pt"
    model = lgb.Booster(model_file=model_path)
    data = calculate("中证200","2022-01-01","2023-01-01")
    data = data.reset_index()
    pre = np.array(data)[:,1:-1]
    res = model.predict(pre)
    data["res"] = res
    data = data[["date","code","res","close"]]
    data["code"]= data["code"].apply(lambda x:"%06d" %x)
    data.sort_values(by=["date", "res"], ascending=[True, False], inplace=True)
    d = data.groupby("date").head(3)
    d.to_csv("res.csv",index=None)
    return d

if __name__ =="__main__":
    test()