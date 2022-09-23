#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
=====================================================================================
@author :  chyao

@file   :  crnn.py

@email  :  chyao7@163.com

@time   :  2020/12/16 17:30

desc    :  ""
=====================================================================================
"""

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

from test01 import calculate


class ModelLgbm:
    def __init__(self,train_name="中证800",train_start="2015-01-01",train_end="2022-01-01",test_name="中证200",test_start="2022-01-01",test_end="2023-01-01",checkpoint="lgbm.pt", result="result.csv"):
        self.train_name = train_name
        self.train_start = train_start
        self.train_end = train_end
        self.test_name = test_name
        self.test_start = test_start
        self.test_end = test_end
        self.save_path = checkpoint
        self.result = result

    def apply_row(self,row):
        # if row['y'] > 2:
        #     return 2
        if row['y'] >0:
            return 1
        # elif row['y'] <-3:
        #     return 1
        else:
            return 0
    
    def train(self):
        data = self.calculate_train(self.train_name,self.train_start,self.train_end)
        data["y"] = data.apply(self.apply_row,axis=1)
        data = np.array(data)
        np.random.shuffle(data)
        self.x = data[:, :-1]
        self.y = data[:, -1]
        x_train,x_test,y_train, y_test = train_test_split(self.x, self.y, test_size=0.2, random_state=1)
        train= lgb.Dataset(data=x_train,label=y_train)
        test = lgb.Dataset(data=x_test, label=y_test)
        parameters = {
            'task': 'train',
            'boosting_type': 'gbdt',  # 设置提升类型
            # 'objective': 'multiclass',
            # 'num_class':3,
            # 'metric':'multi_logloss',# 评估函数
            'objective': 'binary',  # 目标函数
            'metric':{'binary_logloss',"auc"},
            'num_leaves': 8,  # 叶子节点数
            'learning_rate': 0.01,  # 学习速率
            'feature_fraction': 0.9,  # 建树的特征选择比例
            'bagging_fraction': 0.9,  # 建树的样本采样比例
            'bagging_freq': 10,  # k 意味着每 k 次迭代执行bagging
            'verbose': 0,  # <0 显示致命的, =0 显示错误 (警告), >0 显示信息
            "lambda_l1": 0.1,
            "nthread": 16,
            "is_unbalance":"true"

        }
        evals_result = {}
        gbm_model = lgb.train(parameters,
                              train,
                              valid_sets=[train, test],
                              num_boost_round=70000,  # 提升迭代的次数
                              evals_result=evals_result,
                              verbose_eval=500
                              )
        gbm_model.save_model(self.save_path)

    def test(self):
        model = lgb.Booster(model_file=self.save_path)
        data = self.calculate_test(self.test_name,self.test_start,self.test_end)
        data = data.reset_index()
        pre = np.array(data)[:,1:-1]
        res = model.predict(pre)
        data["res"] = res
        data = data[["date","code","res"]]
        data["code"]= data["code"].apply(lambda x:"%06d" %x)
        data.sort_values(by=["date", "res"], ascending=[True, False], inplace=True)
        d = data.groupby("date").head(2)
        d.to_csv(self.result,index=None)
        return d

    def calculate_test(self,name,start,end,):
        @multitasking.task
        @retry(tries=3, delay=1)
        def process(d):
            d.rename(columns = {"日期": "date","股票名称":"name","股票代码":"code","开盘":"open","收盘":"close","最高":"high","最低":"low","成交量":"volume","振幅":"amplitude","换手率":"turnover","涨跌幅":"chg"},  inplace=True)
            d['date'] = pd.to_datetime(d['date'])
            d = d.set_index('date')   
            d1 = d[start:end]
            # d = d['2017-01-01':'2022-01-01']
            d1 = d1.dropna()
            if len(d1)<100:
                return
            # 移动均线
            d1 = self.calculate(d1)
            a = d1.code
            d1 = d1.drop(columns=["code",'Unnamed: 0',"name"])
            # for i in d1.columns:
            #     d1[i]=(d1[i]-d2[i].mean())/d2[i].std()
            d1 = d1.apply(lambda x: (x - np.mean(x)) / (np.std(x)))
            d1["code"] = a
            all.append(d1)            
            return
        all = []
        datas = get_members(name)["股票代码"]
        for code in datas:
            path = os.path.join("./data",code+".csv")
            if not os.path.exists(path):
                continue
            df = pd.read_csv(path)
            process(df)
        multitasking.wait_for_tasks()
        features = pd.concat(all, axis=0)
        return features

    def calculate_train(self,name,start,end,):
        @multitasking.task
        @retry(tries=3, delay=1)
        def process(d):
            d.rename(columns = {"日期": "date","股票名称":"name","股票代码":"code","开盘":"open","收盘":"close","最高":"high","最低":"low","成交量":"volume","振幅":"amplitude","换手率":"turnover","涨跌幅":"chg"},  inplace=True)
            d['date'] = pd.to_datetime(d['date'])
            d = d.set_index('date')
            d = d[start:end]
            # d = d['2017-01-01':'2022-01-01']
            d = d.dropna()
            if len(d)<100:
                return
            # 移动均线
            d = self.calculate(d)
            a = ( d.close.shift(-1)-d.close)/ d.close
            d = d.drop(columns=["code",'Unnamed: 0',"name"])
            d = d.apply(lambda x: (x - np.mean(x)) / (np.std(x)))
            d["y"] = a
            all.append(d)            
            return
        all = []
        datas = get_members(name)["股票代码"]
        for code in datas:
            path = os.path.join("./data",code+".csv")
            if not os.path.exists(path):
                continue
            df = pd.read_csv(path)
            process(df)
        multitasking.wait_for_tasks()
        features = pd.concat(all, axis=0)
        features = features[features["high"]!=features["low"]]
        print(features.head())
        return features

    def calculate(self,d):
        d["close_2"] = (d.close - d.close.shift(2))/ d.close.shift(2)
        # print(d.close.rolling(3).mean())
        d["close_3_m"] = (d.close - d.close.rolling(3).mean())/ d.close.rolling(3).mean()
        d["close_3"] = (d.close - d.close.shift(3))/ d.close.shift(3)
        d["close_7_m"] = (d.close - d.close.rolling(7).mean())/ d.close.rolling(7).mean()
        d["close_7"] = (d.close - d.close.shift(7))/ d.close.shift(7)
        d["close_low"] = (d.close-d.low.shift(1))/d.close
        d["close_low_3"] = (d.close-d.low.shift(3))/d.close
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
        d["ema_30_r"] = d.close/ta.EMA(d.close, timeperiod=30)
        d["ema_10_r"] = d.close/ta.EMA(d.close, timeperiod=10)
        d["ema_5_r"] = d.close/ta.EMA(d.close, timeperiod=5)
        d["ema_3_r"] = d.close/ta.EMA(d.close, timeperiod=5)
        d["ema_30"] = ta.EMA(d.close, timeperiod=30)
        d["ema_10"] = ta.EMA(d.close, timeperiod=10)
        d["ema_5"] = ta.EMA(d.close, timeperiod=5)
        d["ema_3"] = ta.EMA(d.close, timeperiod=5)

        d["adrx_14"] = ta.ADXR(d.high, d.low, d.close, timeperiod=14)
        d["adrx_8"] = ta.ADXR(d.high, d.low, d.close, timeperiod=8)
        d["adrx_5_r"] = d.close/ta.ADXR(d.high, d.low, d.close, timeperiod=5)
        d["adrx_3_r"] = d.close/ta.ADXR(d.high, d.low, d.close, timeperiod=3)
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
        return d

if __name__=="__main__":
    train = ModelLgbm(checkpoint="./checkpoint/labmres.pt",test_name="沪深300")
    # train = ModelLgbm(checkpoint="./checkpoint/labmres.pt",test_name="中证200")

    # train.train()
    train.test()
