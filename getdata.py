from os import cpu_count
import re
from efinance.stock import get_members
from efinance.stock import get_realtime_quotes,get_quote_history
import numpy as np
import pandas as pd
from tqdm import tqdm
import multitasking
multitasking.set_max_threads(multitasking.config["CPU_CORES"])
# print(multitasking.config["CPU_CORES"])
@multitasking.task
def req(code,szcode):
    data = get_quote_history(code,fqt=1) 
    if len(data)<365:
        return
    # print(int(data["鏃ユ湡"].iloc[0].split("-")[0]))
    if  int(data["日期"].iloc[-1].split("-")[0])<2021:
        return
    data = pd.merge(szcode,data,left_on="日期",right_on="日期",how="left")
    data.to_csv(f"./data/{code}.csv",encoding="utf-8")

def getdata(codepool="沪深300"):
    name = "上证指数"
    sz = get_quote_history(name,fqt=1)
    sz.to_csv(f"./data/{name}.csv")
    szcode = sz["日期"]
    # codes = get_realtime_quotes()
    # codes = np.array(codes["鑲＄エ浠ｇ爜"]).tolist()
    codes = get_members(codepool)["股票代码"]
    import time
    for code in tqdm(codes):
        req(code,szcode)


if __name__=="__main__":
    getdata(codepool="中证800")