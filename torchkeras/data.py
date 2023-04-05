from pathlib import Path 
from PIL import Image 
import numpy as np 
import os
import sys 
import time 
from tqdm import tqdm 
import re
import requests
from urllib import parse 

path = Path(__file__)

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

def download_baidu_pictures(keyword,needed_pics_num=100,save_dir=None):
    spider = _BaiduPictures(keyword,needed_pics_num,save_dir)
    spider.run()
     
class _BaiduPictures:
    def __init__(self,keyword,needed_pics_num=100,save_dir=None):
        from fake_useragent import UserAgent 
        self.save_dir = save_dir if save_dir is not None else './{}'.format(keyword)
        self.name_ = keyword
        self.name = parse.quote(self.name_) 
        self.needed_pics_num = needed_pics_num
        self.times = str(int(time.time()*1000)) 
        self.url = 'https://image.baidu.com/search/acjson?tn=resultjson_com&logid=8032920601831512061&ipn=rj&ct=201326592&is=&fp=result&fr=&word={}&cg=star&queryWord={}&cl=2&lm=-1&ie=utf-8&oe=utf-8&adpicid=&st=&z=&ic=&hd=&latest=&copyright=&s=&se=&tab=&width=&height=&face=&istype=&qc=&nc=1&expermode=&nojc=&isAsync=&pn={}&rn=30&gsm=1e&{}='
        self.headers = {'User-Agent':UserAgent().random}

    def get_one_html(self,url,pn):
        response = requests.get(url=url.format(self.name,self.name, pn, self.times), headers=self.headers).content.decode('utf-8')
        return response
    
    def parse_html(self,regex,html):
        content = regex.findall(html)
        return content
    
    def get_two_html(self,url):
        response = requests.get(url=url, headers=self.headers).content
        return response

    def run(self):
        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)
        response = self.get_one_html(self.url,0)
        regex1 = re.compile('"displayNum":(.*?),')
        ori_num = self.parse_html(regex1,response)[0] 
        num = min(int(ori_num),self.needed_pics_num)
        print('{} {} pictures founded. start downloading {} pictures...'.format(
            ori_num,self.name_,num)) 
    
        if int(num)%30 == 0:
            pn = int(int(num)/30)
        else:
            pn = int(int(num)//30 + 2)
        cnt,loop = 0,tqdm(total=num,file=sys.stdout)
        for i in range(pn): 
            try:
                resp = self.get_one_html(self.url, i * 30)
                regex2 = re.compile('"middleURL":"(.*?)"')
                urls = [x for x in self.parse_html(regex2,resp) if x.startswith('http')]
                for u in urls:  
                    try:
                        content = self.get_two_html(u) 
                        img_name = '{}.jpg'.format('0'*max(6-len(str(cnt)),1)+str(cnt))
                        img_path = os.path.join(self.save_dir,img_name)
                        with open(img_path,'wb') as f:
                            f.write(content)
                        cnt+=1
                        loop.update(1)
                        if cnt>=num:
                            break
                    except Exception as err:
                        pass
                else:
                    continue
                break
            except Exception as err:
                print(err,file=sys.stderr)
            time.sleep(1.0) 
        loop.close()
        print('saved {} pictures in dir {}'.format(cnt, self.save_dir))