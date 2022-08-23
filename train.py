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
from ast import main
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
        d = d.drop(columns=["code",'Unnamed: 0',"name"])
        d = d.apply(lambda x: (x - np.mean(x)) / (np.std(x)))
        d["y"] = a*100
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

class Trainer:
    def __init__(self, data, path):
        data["y"] = data.apply(self.apply_row,axis=1)
        data = np.array(data)
        np.random.shuffle(data)
        self.x = data[:, :-1]
        self.y = data[:, -1]
        print(self.y)
        # self.x, self.y = np.array(self.x), np.array(self.y)
        self.save_path = path
    
    def apply_row(self,row):
        # if row['y'] > 2:
        #     return 2
        if row['y'] > 0:
            return 1
        else:
            return 0
    
    def train(self):
        x_train,x_test,y_train, y_test = train_test_split(self.x, self.y, test_size=0.2, random_state=1)

        train= lgb.Dataset(data=x_train,label=y_train)
        test = lgb.Dataset(data=x_test, label=y_test)
        parameters = {
            'task': 'train',
            'boosting_type': 'gbdt',  # 设置提升类型
            'objective': 'binary',  # 目标函数
            'metric':{'binary_logloss', 'auc'}, # 评估函数
            'num_leaves': 8,  # 叶子节点数
            'learning_rate': 0.01,  # 学习速率
            'feature_fraction': 0.9,  # 建树的特征选择比例
            'bagging_fraction': 0.9,  # 建树的样本采样比例
            'bagging_freq': 10,  # k 意味着每 k 次迭代执行bagging
            'verbose': 0,  # <0 显示致命的, =0 显示错误 (警告), >0 显示信息
            "lambda_l1": 0.1,
            "nthread": 16,

        }
        evals_result = {}
        gbm_model = lgb.train(parameters,
                              train,
                              valid_sets=[train, test],
                              num_boost_round=70000,  # 提升迭代的次数
                              evals_result=evals_result,
                              verbose_eval=200
                              )
        gbm_model.save_model(self.save_path)


if __name__=="__main__":
    data = calculate("中证800","2015-01-01","2022-01-01")

    print(len(data))
    train = Trainer(data,"./checkpoint/lgb2.pt")
    train.train()
