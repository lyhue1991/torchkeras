# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import sys
import logging
import numpy as np
import cv2
from pathlib import Path
from PIL import Image 
import argparse

from .logger import set_logging,get_logger
set_logging(name='ocr',verbose=True)
logger = get_logger(name='ocr')


from .textsystem import TextSystem
from .utility import check_img

SUPPORT_DET_MODEL = ['DB']
SUPPORT_REC_MODEL = ['CRNN']

path = Path(__file__)

font_path = str(path.parent.parent/"assets"/"SimHei.ttf")
rec_char_dict_path = str(path.parent.parent/"assets"/"ch_ocr_keys.txt")

def str2bool(v):
    return v.lower() in ("true", "t", "1")

def init_args():
    parser = argparse.ArgumentParser()
    
    # params for model path
    parser.add_argument("--det_model_dir", type=str, default=None)
    parser.add_argument("--cls_model_dir", type=str, default=None)
    parser.add_argument("--rec_model_dir", type=str, default=None)
    
    
    # params for prediction engine
    parser.add_argument("--use_gpu", type=str2bool, default=True)
    parser.add_argument("--use_xpu", type=str2bool, default=False)
    parser.add_argument("--use_npu", type=str2bool, default=False)
    parser.add_argument("--ir_optim", type=str2bool, default=True)
    parser.add_argument("--use_tensorrt", type=str2bool, default=False)
    parser.add_argument("--min_subgraph_size", type=int, default=15)
    parser.add_argument("--precision", type=str, default="fp32")
    parser.add_argument("--gpu_mem", type=int, default=500)
    parser.add_argument("--gpu_id", type=int, default=0)

    # params for text detector
    parser.add_argument("--image_dir", type=str)
    parser.add_argument("--page_num", type=int, default=0)
    parser.add_argument("--det_algorithm", type=str, default='DB')
    parser.add_argument("--det_limit_side_len", type=float, default=960)
    parser.add_argument("--det_limit_type", type=str, default='max')
    parser.add_argument("--det_box_type", type=str, default='quad')

    # DB parmas
    parser.add_argument("--det_db_thresh", type=float, default=0.3)
    parser.add_argument("--det_db_box_thresh", type=float, default=0.6)
    parser.add_argument("--det_db_unclip_ratio", type=float, default=1.5)
    parser.add_argument("--max_batch_size", type=int, default=10)
    parser.add_argument("--use_dilation", type=str2bool, default=False)
    parser.add_argument("--det_db_score_mode", type=str, default="fast")


    # params for text recognizer
    parser.add_argument("--rec_algorithm", type=str, default='CRNN')
    parser.add_argument("--rec_image_inverse", type=str2bool, default=True)
    parser.add_argument("--rec_image_shape", type=str, default="3, 48, 320")
    parser.add_argument("--rec_batch_num", type=int, default=6)
    parser.add_argument("--max_text_length", type=int, default=25)
    parser.add_argument(
        "--rec_char_dict_path",
        type=str,
        default=rec_char_dict_path)
    parser.add_argument("--use_space_char", type=str2bool, default=True)
    parser.add_argument(
        "--vis_font_path", type=str, default=font_path)
    parser.add_argument("--drop_score", type=float, default=0.5)


    # params for text classifier
    parser.add_argument("--use_angle_cls", type=str2bool, default=True)
    parser.add_argument("--cls_image_shape", type=str, default="3, 48, 192")
    parser.add_argument("--label_list", type=list, default=['0', '180'])
    parser.add_argument("--cls_batch_num", type=int, default=6)
    parser.add_argument("--cls_thresh", type=float, default=0.9)

    parser.add_argument("--enable_mkldnn", type=str2bool, default=False)
    parser.add_argument("--cpu_threads", type=int, default=10)
    parser.add_argument("--use_pdserving", type=str2bool, default=False)
    parser.add_argument("--warmup", type=str2bool, default=False)


    parser.add_argument(
        "--draw_img_save_dir", type=str, default="./inference_results")
    parser.add_argument("--save_crop_res", type=str2bool, default=False)
    parser.add_argument("--crop_res_save_dir", type=str, default="./output")

    # multi-process
    parser.add_argument("--use_mp", type=str2bool, default=False)
    parser.add_argument("--total_process_num", type=int, default=1)
    parser.add_argument("--process_id", type=int, default=0)

    parser.add_argument("--benchmark", type=str2bool, default=False)
    parser.add_argument("--save_log_path", type=str, default="./log_output/")

    parser.add_argument("--use_onnx", type=str2bool, default=True)
    return parser


def parse_args(mMain=True):
    import argparse
    parser = init_args()
    parser.add_help = mMain
    parser.add_argument("--lang", type=str, default='ch')
    parser.add_argument("--det", type=str2bool, default=True)
    parser.add_argument("--rec", type=str2bool, default=True)
    parser.add_argument("--type", type=str, default='ocr')
            
    if mMain:
        return parser.parse_args()
    else:
        inference_args_dict = {}
        for action in parser._actions:
            inference_args_dict[action.dest] = action.default
        return argparse.Namespace(**inference_args_dict)
    
    
class Pipeline(TextSystem):
    def __init__(self, 
                 det_model_dir=None,
                 rec_model_dir=None,
                 rec_char_dict_path=None,
                 rec_image_shape=None, #"3, 48, 320"
                 **kwargs):
        
        for k,v in locals().items():
            if v is not None and k in kwargs.keys():
                kwargs[k]=v
        
        params = parse_args(mMain=False)
        params.__dict__.update(**kwargs)

        self.use_angle_cls = params.use_angle_cls
        lang, det_lang = params.lang,params.lang

        if params.rec_char_dict_path is None:
            params.rec_char_dict_path = rec_char_dict_path

        logger.debug(params)
        
        super().__init__(params)
        self.page_num = params.page_num

    def ocr(self, img, det=True, rec=True, cls=True):
        """
        ocr 
        args：
            img: img for ocr, support ndarray, img_path and list and PIL.Image or ndarray
            det: use text detection or not. If false, only rec will be exec. Default is True
            rec: use text recognition or not. If false, only det will be exec. Default is True
            cls: use angle classifier or not. Default is True. If true, the text with rotation of 180 degrees can be recognized. If no text is rotated by 180 degrees, use cls=False to get better performance. Text with rotation of 90 or 270 degrees can be recognized even if cls=False.
        """
        assert isinstance(img, (np.ndarray, list, str, bytes, Image.Image))
        
        if isinstance(img, list) and det == True:
            logger.error('When input a list of images, det must be false')
            exit(0)
        if cls == True and self.use_angle_cls == False:
            logger.warning(
                'Since the angle classifier is not initialized, the angle classifier will not be uesd during the forward process'
            )

        img = check_img(img)
        # for infer pdf file
        if isinstance(img, list):
            if self.page_num > len(img) or self.page_num == 0:
                self.page_num = len(img)
            imgs = img[:self.page_num]
        else:
            imgs = [img]
        if det and rec:
            ocr_res = []
            for idx, img in enumerate(imgs):
                dt_boxes, rec_res, _ = self.__call__(img, cls)
                tmp_res = [[box.tolist(), res]
                           for box, res in zip(dt_boxes, rec_res)]
                ocr_res.append(tmp_res)
            return ocr_res
        elif det and not rec:
            ocr_res = []
            for idx, img in enumerate(imgs):
                dt_boxes, elapse = self.text_detector(img)
                tmp_res = [box.tolist() for box in dt_boxes]
                ocr_res.append(tmp_res)
            return ocr_res
        else:
            ocr_res = []
            cls_res = []
            for idx, img in enumerate(imgs):
                if not isinstance(img, list):
                    img = [img]
                if self.use_angle_cls and cls:
                    img, cls_res_tmp, elapse = self.text_classifier(img)
                    if not rec:
                        cls_res.append(cls_res_tmp)
                rec_res, elapse = self.text_recognizer(img)
                ocr_res.append(rec_res)
            if not rec:
                return cls_res
            return ocr_res
        
    def plot_ocr(self,img,
                 plot_boxes=True,
                 plot_txts=True):

        from . import plots
        assert isinstance(img, (Image.Image,str)), 'img should be a path/link or a PIL Image object!'
        results = self.ocr(img)
        result = results[0]
        boxes = [line[0] for line in result]
        txts = [line[1][0] for line in result]
        scores = [line[1][1] for line in result]
        
        if isinstance(img,Image.Image):
            image = img.copy()
        elif img.startswith('http'):
            import PIL,requests
            image = PIL.Image.open(requests.get(img, stream=True).raw)
            image = PIL.ImageOps.exif_transpose(image)
        elif os.path.exists(img):
            image = Image.open(img)
        else:
            raise Exception('img is not a valid url or path!')
        
        if plot_boxes and plot_txts:
            im_show = plots.draw_ocr(image, boxes, txts, scores)
        elif plot_boxes and not plot_txts:
            im_show = plots.draw_boxes(image,boxes)
        elif not plot_boxes and plot_txts:
            im_show = plots.draw_txts(image, boxes, txts, scores)
        else:
            im_show = image.copy()
        return im_show