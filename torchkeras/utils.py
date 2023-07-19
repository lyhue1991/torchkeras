import torch 
import datetime
from copy import deepcopy
import random
import numpy as np 
import pandas as pd 
from PIL import Image, ImageFont, ImageDraw
from pathlib import Path 
from argparse import Namespace
import os 

def printlog(info):
    nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print("\n"+"=========="*8 + "%s"%nowtime)
    print(info+'...\n\n')
        
def seed_everything(seed=42):
    print(f"Global seed set to {seed}")
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return seed

def delete_object(obj):
    import gc
    obj = None
    gc.collect()
    del obj
    with torch.no_grad():
        torch.cuda.empty_cache()
        
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

def prettydf(df,nrows=20,ncols=20,str_len=9,show=True):
    from prettytable import PrettyTable
    if len(df)>nrows:
        df = df.head(nrows).copy()
        df.loc[len(df)] = '...'
    if len(df.columns)>ncols:
        df = df.iloc[:,:ncols].copy()
        df['...'] = '...'
        
    def fmt(x):
        if isinstance(x, (float,np.float64)):
            return str(round(x,4))
        else:
            s = str(x) if len(str(x))<str_len else str(x)[:str_len-3]+'...'
            for char in ['\n','\r','\t','\v','\b']:
                s = s.replace(char,' ')
            return s
        
    df = df.applymap(fmt)
    table = PrettyTable()
    table.field_names = df.columns.tolist()
    rows =  df.values.tolist()
    table.add_rows(rows)
    if show:
        print(table)
    return table

def text_to_image(text):
    path = Path(__file__)
    simhei = path.parent/"assets/SimHei.ttf"
    lines  = len(text.split("\n")) 
    image = Image.new("RGB", (800, lines*20), (255, 255, 255))
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype(str(simhei),18)
    draw.text((0, 0), text, font=font, fill="#000000")
    return image

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
  

def is_jupyter():
    import contextlib
    with contextlib.suppress(Exception):
        from IPython import get_ipython
        return get_ipython() is not None
    return False

    
  
