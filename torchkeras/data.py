from pathlib import Path 
from PIL import Image 
import numpy as np 
import os
import sys 
import time,datetime 
from tqdm import tqdm 
import re
import requests
from urllib.parse import quote,unquote

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
    
def merge_dataset_folders(from_folders, to_folder, rename_file=True):
    import shutil 
    get_file_num = lambda x: len([y for y in Path(x).rglob('*') if y.is_file()])
    print('before merge:')
    for x in from_folders:
        print(f'{x}: {get_file_num(x)} files')

    done_files = set()
    for i,folder in enumerate(from_folders):
        shutil.copytree(folder,to_folder,dirs_exist_ok=True)
        folder_name = Path(folder).name
        if rename_file:
            files = {x.absolute() for x in Path(to_folder).rglob('*') if x.is_file()}
            todo_files = files - done_files
            for x in  todo_files:
                new_name = folder_name+'_'+str(i)+'_'+x.name
                y = x.rename(x.parent/new_name)
                done_files.add(y)
    print('\nafter merge:')
    print(f'{to_folder}: {get_file_num(to_folder)} files')
    return to_folder 


def get_example_image(img_name='park.jpg'):
    'name can be park.jpg / zidane.jpg'
    img_path = str(path.parent/f"assets/{img_name}")
    assert os.path.exists(img_path), 'img_name can only be  park.jpg / zidane.jpg'
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

def download_github_file(url,save_name=None):
    import torch
    raw_url = url.replace('://github.com/','://raw.githubusercontent.com/').replace('/blob/','/')
    if save_name is None:
        save_name = unquote(os.path.basename(raw_url))
    torch.hub.download_url_to_file(raw_url,save_name)
    print('saved file: '+save_name,file = sys.stderr)
    return save_name


def download_baidu_pictures(keyword,needed_pics_num=100,save_dir=None):
    spider = BaiduPictures(keyword,needed_pics_num,save_dir)
    spider.run()
     
class BaiduPictures:
    def __init__(self,keyword,needed_pics_num=100,save_dir=None):
        from fake_useragent import UserAgent 
        
        self.save_dir = save_dir if save_dir is not None else './{}'.format(keyword)
        self.name_ = keyword
        self.name = quote(self.name_) 
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
            ori_num,self.name_,num),file = sys.stderr) 
    
        if int(num)%30 == 0:
            pn = int(int(num)/30)
        else:
            pn = int(int(num)//30 + 2)
        cnt,loop = 0,tqdm(total=num)
        for i in range(pn): 
            try:
                resp = self.get_one_html(self.url, i * 30)
                regex2 = re.compile('"middleURL":"(.*?)"')
                urls = [x for x in self.parse_html(regex2,resp) if x.startswith('http')]
                for u in urls:  
                    try:
                        content = self.get_two_html(u) 
                        img_name = '{}.jpg'.format(datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
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
        print('saved {} pictures in dir {}'.format(cnt, self.save_dir),file = sys.stderr)
        
        
class ImageCleaner:
    def __init__(self,img_files, work_dir = 'ImageCleaner'):
        
        # img_files can be: 1, a folder; 2, img_path list; 3, folder list; 
        super().__init__()
        import fastdup
        self.img_files = img_files
        self.fd = fastdup.create(work_dir=work_dir)
        
    def run_summary(self, duplicate_similirity =0.99, outlier_percentile=0.02):
        self.fd.run(input_dir=self.img_files, cc_threshold=duplicate_similirity,
                    outlier_percentile=outlier_percentile, overwrite=True)
        return self.fd.summary() 
    
    def get_stats(self):
        return self.fd.img_stats() 
    
    def get_duplicates(self,):
        df = self.fd.connected_components()[0]
        
        dfclusters  = df[['component_id','filename','mean_distance',
           'index']].groupby('component_id').agg({'filename':lambda x:list(x),
            'mean_distance':np.mean,'index':'count'})
        
        dfcluster = dfclusters.reset_index(drop=False)
        dfcluster.columns = ['component_id','files','mean_distance','len']
        dfcluster.index = range(len(dfcluster))
        
        return dfcluster 
    
    def get_outliers(self):
        return self.fd.outliers() 
    
    def delete_duplicates(self,):
        dfcluster = self.get_duplicates()
        for files in tqdm(dfcluster['files']):
            for file in files[1:]:
                os.remove(file)
    
    def delete_outliers(self,):
        dfoutlier = self.fd.outliers()
        for file in tqdm(dfoutlier['filename_outlier']):
            os.remove(file)
            
    def vis_duplicates(self):
        self.fd.vis.duplicates_gallery()
            
    def vis_outliers(self,):
        self.fd.vis.outliers_gallery()
    
    def vis_bright(self):
        self.fd.vis.stats_gallery('bright')
        
    def vis_dark(self):
        self.fd.vis.stats_gallery('dark')
        
    def vis_blur(self):
        self.fd.vis.stats_gallery('blur')
    
    def vis_clusters(self):
        self.fd.vis.component_gallery() 