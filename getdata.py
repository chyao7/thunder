from efinance.stock import get_members
from efinance.stock import get_realtime_quotes,get_quote_history
import numpy as np
import pandas as pd
from tqdm import tqdm
def getdata():
    name = "上证指数"
    szcode = get_quote_history(name,fqt=1)
    szcode.to_csv(f"./data/{name}.csv")
    szcode = szcode["日期"]
    # codes = get_realtime_quotes()
    # codes = np.array(codes["股票代码"]).tolist()
    codes = get_members("中证200")["股票代码"]
    for code in tqdm(codes):
        data = get_quote_history(code,fqt=1) 
        if len(data)<365:
            continue
        print(int(data["日期"].iloc[0].split("-")[0]))
        if  int(data["日期"].iloc[-1].split("-")[0])<2021:
            continue
        data = pd.merge(szcode,data,left_on="日期",right_on="日期",how="left")
        data.to_csv(f"./data/{code}.csv",encoding="utf-8")


if __name__=="__main__":
    getdata()