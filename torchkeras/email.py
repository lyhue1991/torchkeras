# -*- coding: utf-8 -*-
import smtplib
import traceback 
from email.mime.text import MIMEText

def send_msg(receivers,subject,msg=""):
    
    """
    Examples:
    ---------
    >> subject = "info@train_model.py" #邮件主题
    >> msg = "auc=0.98" #邮件内容
    >> receivers = ["265011xxxx@qq.com"] #收件人
    >> send_msg(receivers,subject,msg)
    """
    
    #设置服务器所需信息
    mail_host = 'smtp.yeah.net'  
    mail_user = 'bugrobot'  
    mail_pass = 'NPWPJBSIVXRTYUOB'   #密码(部分邮箱为授权码) 
    sender = 'bugrobot@yeah.net'  

    #构造邮件内容
    message = MIMEText(msg,'plain','utf-8')  
    message['Subject'] = subject
    message['From'] = sender     
    message['To'] = receivers[0]  

    #登录并发送邮件
    try:
        smtpObj = smtplib.SMTP() 
        #连接到服务器
        smtpObj.connect(mail_host,25)
        #登录到服务器
        smtpObj.login(mail_user,mail_pass) 
        #发送
        smtpObj.sendmail(
            sender,receivers,message.as_string()) 
        #退出
        smtpObj.quit() 
        return 'send_msg success'
    except smtplib.SMTPException as e:
        error = 'send_msg error : '+str(e)
        print(error)
        return error

def monitor_run(function,receivers):
    """
    Examples:
    ---------
    >> receivers = ["265011xxxx@qq.com"] #收件人
    >> def f():
    >>    return 1/0
    >> monitor_run(f,receivers)
    """
    try:
        function()
    except Exception as e:
        error_msg = traceback.format_exc()
        send_msg(receivers,"error@"+function.__name__,error_msg)
        raise e