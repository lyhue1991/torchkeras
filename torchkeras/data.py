from pathlib import Path 
from PIL import Image 
import os
import numpy as np 
path = Path(__file__)

def get_example_image(img_name='park.jpg'):
    'name can be bus.jpg / park.jpg / zidane.jpg'
    img_path = str(path.parent/f"assets/{img_name}")
    assert os.path.exists(img_path), 'img_name can only be bus.jpg / park.jpg / zidane.jpg'
    return Image.open(img_path)

def get_url_img(url):
    from skimage import io
    arr = io.imread(url)
    return Image.fromarray(arr)

def download_image(url):
    import PIL,requests
    image = PIL.Image.open(requests.get(url, stream=True).raw)
    image = PIL.ImageOps.exif_transpose(image)
    return image

def resize_and_pad_image(image,width,height):
    img = image
    w, h = img.size 
    ratio = w / float(h)
    imgH,imgW = height,width
    
    if imgH * ratio >= imgW:
        resized_w = imgW
        resized_h =  np.floor(imgW/ratio).astype('int32')
        resized_image = img.resize((resized_w, resized_h))
        resized_arr = np.array(resized_image).astype('float32')

        if img.mode=='L':
            padding_arr = np.zeros((imgH, imgW), dtype=np.float32)
        else:
            assert img.mode=='RGB'
            padding_arr = np.zeros((imgH, imgW,3), dtype=np.float32)
            
        padding_arr[:resized_h, :] = resized_arr
        padding_im = Image.fromarray(padding_arr.astype(np.uint8))
        box_factor = resized_h/h
        return  padding_im
    else:
        resized_h = imgH
        resized_w = np.floor(imgH*ratio).astype('int32')
        resized_image = img.resize((resized_w, resized_h))
        resized_arr = np.array(resized_image).astype('float32')
        
        if img.mode=='L':
            padding_arr = np.zeros((imgH, imgW), dtype=np.float32)
        else:
            assert img.mode=='RGB'
            padding_arr = np.zeros((imgH, imgW,3), dtype=np.float32)
            
        padding_arr[:, :resized_w] = resized_arr
        padding_im = Image.fromarray(padding_arr.astype(np.uint8))
        box_factor = resized_w/w
        return padding_im
    
    

   
    