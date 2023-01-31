import torch 
import datetime
from copy import deepcopy
import numpy as np 
from PIL import Image, ImageFont, ImageDraw
import pathlib
from argparse import Namespace

def text_to_image(text):
    path = pathlib.Path(__file__)
    simhei = path.parent/"SimHei.ttf"
    lines  = len(text.split("\n")) 
    image = Image.new("RGB", (800, lines*20), (255, 255, 255))
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype(str(simhei),18)
    draw.text((0, 0), text, font=font, fill="#000000")
    return image

def image_to_tensor(image):
    from torchvision.transforms import ToTensor
    tensor = ToTensor()(np.array(image))
    return tensor

def namespace2dict(namespace):
    result = {}
    for k,v in vars(namespace).items():
        if not isinstance(v,Namespace):
            result[k] = v
        else:
            v_dic = namespace2dict(v)
            for v_key,v_value in v_dic.items():
                result[k+"."+v_key] = v_value
    return result 

def colorful(obj,color="red", display_type="plain"):
    # 彩色输出格式：
    # 设置颜色开始 ：\033[显示方式;前景色;背景色m
    # 说明：
    # 前景色            背景色           颜色
    # ---------------------------------------
    # 30                40              黑色
    # 31                41              红色
    # 32                42              绿色
    # 33                43              黃色
    # 34                44              蓝色
    # 35                45              紫红色
    # 36                46              青蓝色
    # 37                47              白色
    # 显示方式           意义
    # -------------------------
    # 0                终端默认设置
    # 1                高亮显示
    # 4                使用下划线
    # 5                闪烁
    # 7                反白显示
    # 8                不可见
    color_dict = {"black":"30", "red":"31", "green":"32", "yellow":"33",
                    "blue":"34", "purple":"35","cyan":"36",  "white":"37"}
    display_type_dict = {"plain":"0","highlight":"1","underline":"4",
                "shine":"5","inverse":"7","invisible":"8"}
    s = str(obj)
    color_code = color_dict.get(color,"")
    display  = display_type_dict.get(display_type,"")
    out = '\033[{};{}m'.format(display,color_code)+s+'\033[0m'
    return out 

def get_call_file(): 
    import traceback
    stack = traceback.extract_stack()
    return stack[-2].filename 

def getNotebookPath():
    from jupyter_server import serverapp
    from jupyter_server.utils import url_path_join
    from pathlib import Path
    import requests,re
    kernelIdRegex = re.compile(r"(?<=kernel-)[\w\d\-]+(?=\.json)")
    kernelId = kernelIdRegex.search(get_ipython().config["IPKernelApp"]["connection_file"])[0]
    for jupServ in serverapp.list_running_servers():
        for session in requests.get(url_path_join(jupServ["url"], "api/sessions"),
                                    params={"token":jupServ["token"]}).json():
            if kernelId == session["kernel"]["id"]:
                return str(Path(jupServ["root_dir"]) / session["notebook"]['path']) 
    raise Exception('failed to get current notebook path')
    
def plot_metric(dfhistory, metric):
    import plotly.graph_objs as go
    train_metrics = dfhistory["train_"+metric].values.tolist()
    val_metrics = dfhistory['val_'+metric].values.tolist()
    epochs = list(range(1, len(train_metrics) + 1))
    
    train_scatter = go.Scatter(x = epochs, y=train_metrics, mode = "lines+markers",
                               name = 'train_'+metric,marker = dict(size=8,color="blue"),
                                line= dict(width=2,color="blue",dash="dash"))
    val_scatter = go.Scatter(x = epochs, y=val_metrics, mode = "lines+markers",
                            name = 'val_'+metric,marker = dict(size=10,color="red"),
                            line= dict(width=2,color="red",dash="solid"))
    fig = go.Figure(data = [train_scatter,val_scatter])
    return fig    


