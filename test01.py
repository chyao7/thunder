from fileinput import close
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
        # d.rename(columns = {"日期": "date","股票名称":"name","股票代码":"code","开盘":"open","收盘":"close","最高":"high","最低":"low","成交量":"volume"},  inplace=True)
        # d.rename(columns = {"日期": "date","股票名称":"name","股票代码":"code","开盘":"open","收盘":"close","最高":"high","最低":"low","成交量":"volume","振幅":"amplitude","换手率":"turnover"},  inplace=True)
        d.rename(columns = {"日期": "date","股票名称":"name","股票代码":"code","开盘":"open","收盘":"close","最高":"high","最低":"low","成交量":"volume","振幅":"amplitude","换手率":"turnover","涨跌幅":"chg"},  inplace=True)
        
        d['date'] = pd.to_datetime(d['date'])
        d = d.set_index('date')
        d = d[start:end]
        d = d.dropna()
        if len(d)<100:
            return
        d["close_2"] = (d.close - d.close.shift(2))/ d.close.shift(2)

        d["close_3_m"] = (d.close - d.close.rolling[3].mean())/ d.close.rolling[3].mean()
        d["close_3"] = (d.close - d.close.shift(3))/ d.close.shift(3)
        
        d["close_7_m"] = (d.close - d.close.rolling[7].mean())/ d.close.rolling[7].mean()
        d["close_7"] = (d.close - d.close.shift(7))/ d.close.shift(7)
        
        d["close_low"] = (d.close-d.low.shift(1))/d.clsoe
        d["close_low_3"] = (d.close-d.low.shift(3))/d.clsoe

        d["chg_3"] =  d.chg.rolling(3).mean()
        d["chg_5"] =  d.chg.rolling(5).mean()
        d["chg_7"] =  d.chg.rolling(7).mean()
        d["chg_15"] =  d.chg.rolling(15).mean()        
        d["chg_3_r"] = d.chg.rolling(3).mean()/d.chg.rolling(15).mean()
        d["chg_3_r"] = d.chg/d.chg.rolling(3).mean()

        d["amplitude_3"] = d.amplitude.rolling(3).mean()
        d["amplitude_7"] = d.amplitude.rolling(7).mean()
        d["amplitude_3_r"] = d.amplitude/d.amplitude.rolling(3).mean()
        d["amplitude_7_r"] = d.amplitude/d.amplitude.rolling(7).mean()

        d["turnover_3"] = d.turnover.rolling(3).mean()
        d["turnover_7"] = d.turnover.rolling(7).mean()
        d["turnover_3_r"] = d.turnover/d.turnover.rolling(3).mean()
        d["turnover_7_r"] = d.turnover/d.turnover.rolling(7).mean()

        d["volume_2"] = d.volume.rolling(2).mean()
        d["volume_4"] = d.volume.rolling(4).mean()
        d["volume_6"] = d.volume.rolling(6).mean()

        d["volume_2_r"] = d.volume/d.volume.rolling(2).mean()
        d["volume_4_r"] = d.volume/d.volume.rolling(4).mean()
        d["volume_6_r"] = d.volume/d.volume.rolling(6).mean()

        # 移动均线
        d["ema_30"] = ta.EMA(d.close, timeperiod=30)
        d["ema_10"] = ta.EMA(d.close, timeperiod=10)
        d["ema_5"] = ta.EMA(d.close, timeperiod=5)
        d["ema_3"] = ta.EMA(d.close, timeperiod=5)
        d["adrx_14"] = ta.ADXR(d.high, d.low, d.close, timeperiod=14)
        d["adrx_8"] = ta.ADXR(d.high, d.low, d.close, timeperiod=8)
        d["adrx_5"] = ta.ADXR(d.high, d.low, d.close, timeperiod=5)
        d["adrx_3"] = ta.ADXR(d.high, d.low, d.close, timeperiod=3)
        d["apo"] = ta.APO(d.close, fastperiod=5, slowperiod=20, matype=0)
        d["apo_37"] = ta.APO(d.close, fastperiod=3, slowperiod=7, matype=0)
        d["apo_37"] = ta.APO(d.close, fastperiod=3, slowperiod=15, matype=0)
        d["CCI_14"] = ta.CCI(d.high, d.low, d.close, timeperiod=14)
        d["CCI_7"] = ta.CCI(d.high, d.low, d.close, timeperiod=7)
        d["CCI_4"] = ta.CCI(d.high, d.low, d.close, timeperiod=4)
        d["MFI_10"] = ta.MFI(d.high, d.low, d.close, d.volume, timeperiod=10)
        d["MFI_3"] = ta.MFI(d.high, d.low, d.close, d.volume, timeperiod=3)
        d["PLUS_DI_10"] = ta.PLUS_DI(d.high, d.low, d.close, timeperiod=10)
        d["PLUS_DI_6"] = ta.PLUS_DI(d.high, d.low, d.close, timeperiod=10)
        d["PLUS_DI_3"] = ta.PLUS_DI(d.high, d.low, d.close, timeperiod=3)
        d["RSI_14"]=ta.RSI(d.close, timeperiod=14)
        d["RSI_5"]=ta.RSI(d.close, timeperiod=5)
        d["RSI_3"]=ta.RSI(d.close, timeperiod=3)
        d["ROC_14"] = ta.ROC(d.close, timeperiod=14)
        d["ROC_7"] = ta.ROC(d.close, timeperiod=7)
        d["ROC_3"] = ta.ROC(d.close, timeperiod=3)
        d["ADOSC_310"] = ta.ADOSC(d.high, d.low, d.close, d.volume, fastperiod=3, slowperiod=12)
        d["ADOSC_36"] = ta.ADOSC(d.high, d.low, d.close, d.volume, fastperiod=3, slowperiod=6)
        d["ADOSC_10"] = ta.ADOSC(d.high, d.low, d.close, d.volume, fastperiod=10, slowperiod=30)
        d["HT_DCPERIOD"] = ta.HT_DCPERIOD(d.close)
        d["NATR_14"] = ta.NATR(d.high, d.low, d.close, timeperiod=14)
        d["NATR_7"] = ta.NATR(d.high, d.low, d.close, timeperiod=7)
        d["NATR_3"] = ta.NATR(d.high, d.low, d.close, timeperiod=3)
        a = d.code
        d = d.drop(columns=["code",'Unnamed: 0',"name"])
        print(d.head())
        
        d = d.apply(lambda x: (x - np.mean(x)) / (np.std(x)))
        print(d.head())
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
    model_path = "./checkpoint/lgb5.pt"
    model = lgb.Booster(model_file=model_path)
    data = calculate("中证200","2022-01-01","2023-01-01")
    data = data.reset_index()
    pre = np.array(data)[:,1:-1]
    res = model.predict(pre)
    print(res)
    data["res"] = res
    data = data[["date","code","res","close"]]
    data["code"]= data["code"].apply(lambda x:"%06d" %x)
    data.sort_values(by=["date", "res"], ascending=[True, False], inplace=True)
    d = data.groupby("date").head(1)
    d.to_csv("res.csv",index=None)
    return d

if __name__ =="__main__":
    test()