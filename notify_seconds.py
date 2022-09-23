
from getdata import getdata
from test01 import test
from backtest import backtest
import smtplib
from email.mime.text import MIMEText
from email.header import Header
import time
import datetime
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
    data = test()
    today = time.strftime("%Y-%m-%d", time.localtime())
    data = data[data["date"]==today]
    print(data)
    rec = []
    for i,cont in data.iterrows():
        if cont.res>0.85:
            rec.append(f"{cont.code}_{cont.res}_{cont.close}")     
    if rec:
        recnews = "\n".join(rec)
        print(recnews)
        email("system","chyao",f"{today}-recommend",recnews)


if __name__ == '__main__':
    import schedule
    schedule.every(600).seconds.do(notified)
    # schedule.every().day.at("20:30").do(notified)
    # schedule.every().day.at("14:50").do(notified)
    d_time = datetime.datetime.strptime(str(datetime.datetime.now().date()) + '9:30', '%Y-%m-%d%H:%M')
    d_time1 = datetime.datetime.strptime(str(datetime.datetime.now().date()) + '11:30', '%Y-%m-%d%H:%M')
    d_time2 = datetime.datetime.strptime(str(datetime.datetime.now().date()) + '13:00', '%Y-%m-%d%H:%M')
    d_time3 = datetime.datetime.strptime(str(datetime.datetime.now().date()) + '15:00', '%Y-%m-%d%H:%M')
    # 当前时间
    while True:
        n_time = datetime.datetime.now()

        if (n_time > d_time and n_time < d_time1) or (n_time > d_time2 and n_time < d_time3) : 
            schedule.run_pending()
        time.sleep(1)
