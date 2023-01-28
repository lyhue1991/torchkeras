# -*- coding: utf-8 -*-
import torch 
import pytorch_lightning as pl 
from torch.utils.tensorboard import SummaryWriter 
from torchvision.transforms import ToTensor

import datetime
from copy import deepcopy
import numpy as np 
from PIL import Image, ImageFont, ImageDraw
import pathlib
from argparse import Namespace
from .summary import summary
from .utils import text_to_image,image_to_tensor,namespace2dict

class TensorBoardCallback(pl.callbacks.Callback):
    def __init__(self, save_dir = "tb_logs", model_name="default", 
                 log_weight=True, log_weight_freq=5,
                 log_graph=True, example_input_array=None,
                 log_hparams=True, hparams_dict = None) -> None:
        super().__init__()
        self.logger = pl.loggers.TensorBoardLogger(save_dir,model_name)
        self.writer = self.logger.experiment
        self.log_graph = log_graph
        self.log_weight = log_weight
        self.log_weight_freq = log_weight_freq
        self.example_input_array = example_input_array
        self.log_hparams = log_hparams
        self.hparams_dict = namespace2dict(hparams_dict) if  isinstance(hparams_dict,Namespace) else hparams_dict
        
        
    def on_fit_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        """Called when fit begins."""
        
        #weight日志
        if self.log_weight:
            for name, param in pl_module.net.named_parameters():
                self.writer.add_histogram(name, param.clone().cpu().data.numpy(), -1)
            self.writer.flush()
        
        #graph日志
        if not self.log_graph:
            return
        if self.example_input_array is None and pl_module.example_input_array is not None:
            self.example_input_array = pl_module.example_input_array
        if self.example_input_array is None:
            raise Exception("example_input_array needed for graph logging ...")
        net_cpu = deepcopy(pl_module.net).cpu()
        self.writer.add_graph(net_cpu,
            input_to_model = [self.example_input_array])
        self.writer.flush()
        
        #image日志
        #summary_text =  summary(net_cpu,input_data = self.example_input_array)
        #summary_tensor = image_to_tensor(text_to_image(summary_text))
        #self.writer.add_image('summary',summary_tensor,global_step=-1)
        #self.writer.flush()
        del(net_cpu)
        
        
    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        epoch = pl_module.current_epoch
        
        #metric日志
        dic = deepcopy(pl_module.history[epoch]) 
        dic.pop("epoch",None)
        metrics_group = {}
        for key,value in dic.items():
            g = key.replace("train_",'').replace("val_",'')
            metrics_group[g] = dict(metrics_group.get(g,{}), **{key:value})
        for group,metrics in metrics_group.items():
            self.writer.add_scalars(group, metrics, epoch)
        self.writer.flush()
        
        #weight日志
        if self.log_weight and (epoch+1)%self.log_weight_freq==0:
            for name, param in pl_module.net.named_parameters():
                self.writer.add_histogram(name, param.clone().cpu().data.numpy(), epoch)
            self.writer.flush()
        
    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        epoch = pl_module.current_epoch
        dic = deepcopy(pl_module.history[epoch]) 
        if epoch==0 and 'train_loss' not in dic:
            return 
        dic.pop("epoch",None)
        metrics_group = {}
        for key,value in dic.items():
            g = key.replace("train_",'').replace("val_",'')
            metrics_group[g] = dict(metrics_group.get(g,{}), **{key:value})
        for group,metrics in metrics_group.items():
            self.writer.add_scalars(group, metrics, epoch)
        self.writer.flush()
    
    def on_exception(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", exception: BaseException) -> None:
        """Called when any trainer execution is interrupted by an exception."""
        #weight日志
        if self.log_weight:
            for name, param in pl_module.net.named_parameters():
                self.writer.add_histogram(name, param.clone().cpu().data.numpy(), pl_module.current_epoch)
            self.writer.flush()
        self.writer.close()
    
    def on_fit_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        """Called when fit ends."""
        #weight日志
        if self.log_weight:
            for name, param in pl_module.net.named_parameters():
                self.writer.add_histogram(name, param.clone().cpu().data.numpy(), pl_module.current_epoch)
            self.writer.flush()
            
        #hparams日志
        if self.log_hparams:
            hyper_dic = {
                    "version":self.logger.version, 
                    "version_time":datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                }
            if self.hparams_dict is not None:
                hyper_dic.update(self.hparams_dict)
                for k,v in self.hparams_dict.items():
                    if not isinstance(v,(int,float,str,torch.Tensor)):
                        hyper_dic[k] = str(v)
                
                
            dfhistory = pl_module.get_history() 
            monitor = trainer.checkpoint_callback.monitor 
            mode = trainer.checkpoint_callback.mode  
            best_idx = dfhistory[monitor].argmax() if mode=="max" else dfhistory[monitor].argmin()
            metric_dic = dict(dfhistory[[col for col in dfhistory.columns 
                                       if col.startswith("val_")]].iloc[best_idx])  
            
            self.writer.add_hparams(hyper_dic,metric_dic)
            self.writer.flush() 
            
        self.writer.close()


