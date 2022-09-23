import requests

def send_wechat(msg):

    token = "622bb2d49b05429496184e9b9a399d34"#前边复制到那个token
    title = 'today recommend'
    content = msg
    template = 'html'
    url = f"https://www.pushplus.plus/send?token={token}&title={title}&content={content}&template={template}"
    r = requests.get(url=url)
    print("微信消息发送成功")
if __name__=="__main__":
    recnews = "\n".join(["1","2","3"]) 
    send_wechat(recnews)
