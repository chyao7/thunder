
from getdata import getdata
from test import test
from backtest import backtest
import smtplib
from email.mime.text import MIMEText
from email.header import Header
import time
from modelLgbm2 import ModelLgbm
import pandas as pd
from utils.wechart import send_wechat

def email(addresser, recipient,subject, content):
    sender = '1318930007@qq.com'
    receivers = ['chyao7@163.com']  # 接收邮件，可设置为你的QQ邮箱或者其他邮箱

    # 三个参数：第一个为文本内容，第二个 plain 设置文本格式，第三个 utf-8 设置编码
    message = MIMEText(content, 'plain', 'utf-8')
    message['From'] = Header(addresser, 'utf-8')  # 发送者
    message['To'] = Header(recipient, 'utf-8')  # 接收者
    message['Subject'] = Header(subject, 'utf-8')

    try:
        smtpObj = smtplib.SMTP('smtp.qq.com')
        smtpObj .starttls()
        smtpObj .login(sender, "fblazwjumgdvhfej")
        smtpObj.sendmail(sender, receivers, message.as_string())
        print("邮件发送成功")
    except smtplib.SMTPException:
        print("Error: 无法发送邮件")

def notified():
    getdata()
    model = ModelLgbm(checkpoint="./checkpoint/labmres.pt",test_name="沪深300")
    # train.train()
    model.test()

    data = pd.read_csv("./result.csv")
    today = time.strftime("%Y-%m-%d", time.localtime())
    print(today)
    data = data[data["date"]==today]

    rec = []
    for i,cont in data.iterrows():
        rec.append(f"{cont.code}_{cont.res}")
    recnews = "\n".join(rec)
    print(recnews)
    email("system","chyao",f"{today}-recommend",recnews)
    send_wechat(recnews)
    # send_wechat("sssssssssss")

if __name__ == '__main__':
    import schedule
    notified()
    schedule.every().day.at("20:30").do(notified)
    schedule.every().day.at("14:49").do(notified)
    schedule.every().day.at("14:54").do(notified)

    while True:
        schedule.run_pending()
        time.sleep(60)
