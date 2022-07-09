import torch 
import datetime
from copy import deepcopy
import numpy as np 
from PIL import Image, ImageFont, ImageDraw
import pathlib
from torchvision.transforms import ToTensor
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