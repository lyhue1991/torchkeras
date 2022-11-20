import os,sys,time
import numpy as np
import pandas as pd
import datetime 
from tqdm import tqdm 
from copy import deepcopy

import torch
from torch import nn 
from torch.utils.data import DataLoader, Dataset

from accelerate import Accelerator
from accelerate.utils import set_seed
from accelerate import notebook_launcher

#================warm up device(mps/gpu/tpu)======================
import os
from accelerate.utils import write_basic_config
write_basic_config() 

def warm_loop(mixed_precision="no"):
    set_seed(10)
    accelerator = Accelerator(mixed_precision=mixed_precision)
    accelerator.print("device ",accelerator.device, " will be used!")
    
args = ("no",)
notebook_launcher(warm_loop, args, num_processes=1)

#===========================================================================

    
def colorful(obj,color="red", display_type="plain"):
    color_dict = {"black":"30", "red":"31", "green":"32", "yellow":"33",
                    "blue":"34", "purple":"35","cyan":"36",  "white":"37"}
    display_type_dict = {"plain":"0","highlight":"1","underline":"4",
                "shine":"5","inverse":"7","invisible":"8"}
    s = str(obj)
    color_code = color_dict.get(color,"")
    display  = display_type_dict.get(display_type,"")
    out = '\033[{};{}m'.format(display,color_code)+s+'\033[0m'
    return out 

class StepRunner:
    def __init__(self, net, loss_fn, accelerator, stage = "train", metrics_dict = None, 
                 optimizer = None, lr_scheduler = None
                 ):
        self.net,self.loss_fn,self.metrics_dict,self.stage = net,loss_fn,metrics_dict,stage
        self.optimizer,self.lr_scheduler = optimizer,lr_scheduler
        self.accelerator = accelerator
    
    def __call__(self, batch):
        features,labels = batch 
        
        #loss
        preds = self.net(features)
        loss = self.loss_fn(preds,labels)

        #backward()
        if self.optimizer is not None and self.stage=="train":
            self.accelerator.backward(loss)
            self.optimizer.step()
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
            self.optimizer.zero_grad()
            
        
        #all_preds,all_labels = self.accelerator.gather_for_metrics((preds, labels))
        all_preds = self.accelerator.gather(preds)
        all_labels = self.accelerator.gather(labels)
        all_loss = self.accelerator.gather(loss).sum()
            
        #metrics
        step_metrics = {self.stage+"_"+name:metric_fn(all_preds, all_labels).item() 
                        for name,metric_fn in self.metrics_dict.items()}
        
        return all_loss.item(),step_metrics


class EpochRunner:
    def __init__(self,steprunner):
        self.steprunner = steprunner
        self.stage = steprunner.stage
        self.steprunner.net.train() if self.stage=="train" else self.steprunner.net.eval()
        self.accelerator = self.steprunner.accelerator
        
    def __call__(self,dataloader):
        total_loss,step = 0,0
        
        loop = tqdm(enumerate(dataloader), 
                    total =len(dataloader),
                    file=sys.stdout,
                    disable=not self.accelerator.is_local_main_process,
                    ncols = 100
                   )
        
        for i, batch in loop: 
            if self.stage=="train":
                loss, step_metrics = self.steprunner(batch)
            else:
                with torch.no_grad():
                    loss, step_metrics = self.steprunner(batch)
                    
            step_log = dict({self.stage+"_loss":loss},**step_metrics)

            total_loss += loss
            step+=1
            if i!=len(dataloader)-1:
                loop.set_postfix(**step_log)
            else:
                epoch_loss = total_loss/step
                epoch_metrics = {self.stage+"_"+name:metric_fn.compute().item() 
                                 for name,metric_fn in self.steprunner.metrics_dict.items()}
                epoch_log = dict({self.stage+"_loss":epoch_loss},**epoch_metrics)
                loop.set_postfix(**epoch_log)

                for name,metric_fn in self.steprunner.metrics_dict.items():
                    metric_fn.reset()
        return epoch_log
    

class KerasModel(torch.nn.Module):
    def __init__(self,net,loss_fn,metrics_dict=None,optimizer=None,lr_scheduler = None):
        super().__init__()
        self.net,self.loss_fn = net, loss_fn
        self.metrics_dict = nn.ModuleDict(metrics_dict) 
        
        self.optimizer = optimizer if optimizer is not None else torch.optim.Adam(
            net.parameters(), lr=1e-2)
        self.lr_scheduler = lr_scheduler

    def forward(self, x):
        if self.net:
            return self.net.forward(x)
        else:
            raise NotImplementedError

            
    def fit(self, train_data, val_data=None, epochs=10,ckpt_path='checkpoint.pt',
            patience=5, monitor="val_loss", mode="min", mixed_precision='no'):
        
        accelerator = Accelerator(mixed_precision=mixed_precision)
        device = str(accelerator.device)
        device_type = '🐌'  if 'cpu' in device else '⚡️'
        accelerator.print(colorful("<<<<<< "+device_type +" "+ device +" is used >>>>>>"))
        
        optimizer = torch.optim.Adam(self.net.parameters(), lr=1e-2)
    
        net,optimizer,lr_scheduler= accelerator.prepare(
            self.net,self.optimizer,self.lr_scheduler)
        
        train_dataloader,val_dataloader = accelerator.prepare(train_data,val_data)
        
        loss_fn = self.loss_fn
        
        if isinstance(loss_fn,torch.nn.Module):
            loss_fn.to(accelerator.device)
        
        metrics_dict = self.metrics_dict 
        metrics_dict.to(accelerator.device)
        
        history = {}

        for epoch in range(1, epochs+1):

            nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            accelerator.print("\n"+"=========="*8 + "%s"%nowtime)
            accelerator.print("Epoch {0} / {1}".format(epoch, epochs)+"\n")

            # 1，train -------------------------------------------------  
            train_step_runner = StepRunner(
                    net = net,
                    loss_fn = loss_fn,
                    accelerator = accelerator,
                    stage="train",
                    metrics_dict=deepcopy(metrics_dict),
                    optimizer = optimizer,
                    lr_scheduler = lr_scheduler
            )

            train_epoch_runner = EpochRunner(train_step_runner)

            train_metrics = train_epoch_runner(train_dataloader)

            for name, metric in train_metrics.items():
                history[name] = history.get(name, []) + [metric]

            # 2，validate -------------------------------------------------
            if val_dataloader:
                val_step_runner = StepRunner(
                    net = net,
                    loss_fn = loss_fn,
                    accelerator = accelerator,
                    stage="val",
                    metrics_dict= deepcopy(metrics_dict)
                )
                val_epoch_runner = EpochRunner(val_step_runner)
                with torch.no_grad():
                    val_metrics = val_epoch_runner(val_dataloader)

                val_metrics["epoch"] = epoch
                for name, metric in val_metrics.items():
                    history[name] = history.get(name, []) + [metric]

            # 3，early-stopping -------------------------------------------------
            accelerator.wait_for_everyone()
            arr_scores = history[monitor]
            best_score_idx = np.argmax(arr_scores) if mode=="max" else np.argmin(arr_scores)

            if best_score_idx==len(arr_scores)-1:
                unwrapped_net = accelerator.unwrap_model(net)
                accelerator.save(unwrapped_net.state_dict(),ckpt_path)
                accelerator.print(colorful("<<<<<< reach best {0} : {1} >>>>>>".format(monitor,
                     arr_scores[best_score_idx])))

            if len(arr_scores)-best_score_idx>patience:
                accelerator.print(colorful("<<<<<< {} without improvement in {} epoch, early stopping >>>>>>".format(
                    monitor,patience)))
                break 
                
        if accelerator.is_local_main_process:
            self.net.load_state_dict(torch.load(ckpt_path))
            dfhistory = pd.DataFrame(history)
            accelerator.print(dfhistory)
            return dfhistory 
    
    @torch.no_grad()
    def evaluate(self, val_data):
        accelerator = Accelerator()
        self.net = accelerator.prepare(self.net)
        val_data = accelerator.prepare(val_data)
        if isinstance(self.loss_fn,torch.nn.Module):
            self.loss_fn.to(accelerator.device)
        self.metrics_dict.to(accelerator.device)
        
        val_step_runner = StepRunner(net = self.net,stage="val",
                    loss_fn = self.loss_fn,metrics_dict=deepcopy(self.metrics_dict),
                    accelerator = accelerator)
        val_epoch_runner = EpochRunner(val_step_runner)
        val_metrics = val_epoch_runner(val_data)
        return val_metrics
    
    @torch.no_grad()
    def predict(self, dataloader):
        accelerator = Accelerator()
        self.net = accelerator.prepare(self.net)
        dataloader = accelerator.prepare(dataloader)
        self.net.eval()
        result = torch.cat([self.net(t[0]) for t in dataloader])
        return result.data
    
def train_model(create_dataloaders, create_net, create_loss_fn, 
                create_optimizer, create_metrics_dict, 
                create_lr_scheduler = None, 
                epochs=10, ckpt_path='checkpoint.pt',
                patience=5, monitor="val_loss", mode="min",
                mixed_precision = "no"):
    
    accelerator = Accelerator(mixed_precision=mixed_precision)
    device = str(accelerator.device)
    device_type = '🐌'  if 'cpu' in device else '⚡️'
    accelerator.print(colorful("<<<<<< "+device_type +" "+ device +" is used >>>>>>"))
    
    train_dataloader,val_dataloader = create_dataloaders()
    net = create_net()
    loss_fn = create_loss_fn()
    optimizer = create_optimizer(net)
    metrics_dict = nn.ModuleDict(create_metrics_dict())
    lr_scheduler = create_lr_scheduler(optimizer) if create_lr_scheduler is not None else None

    net,optimizer,train_dataloader,val_dataloader,lr_scheduler = accelerator.prepare(
            net,optimizer,train_dataloader,val_dataloader,lr_scheduler)
    
    if isinstance(loss_fn,torch.nn.Module):
        loss_fn.to(accelerator.device)
    
    metrics_dict.to(accelerator.device)
   
    history = {}

    for epoch in range(1, epochs+1):
        
        nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        accelerator.print("\n"+"=========="*8 + "%s"%nowtime)
        accelerator.print("Epoch {0} / {1}".format(epoch, epochs)+"\n")

        # 1，train -------------------------------------------------  
        train_step_runner = StepRunner(
                net = net,
                loss_fn = loss_fn,
                accelerator = accelerator,
                stage="train",
                metrics_dict=deepcopy(metrics_dict),
                optimizer = optimizer,
                lr_scheduler = lr_scheduler
        )
        
        train_epoch_runner = EpochRunner(train_step_runner)
        
        train_metrics = train_epoch_runner(train_dataloader)

        for name, metric in train_metrics.items():
            history[name] = history.get(name, []) + [metric]

        # 2，validate -------------------------------------------------
        if val_dataloader:
            val_step_runner = StepRunner(
                net = net,
                loss_fn = loss_fn,
                accelerator = accelerator,
                stage="val",
                metrics_dict=deepcopy(metrics_dict)
            )
            val_epoch_runner = EpochRunner(val_step_runner)
            with torch.no_grad():
                val_metrics = val_epoch_runner(val_dataloader)
            
            val_metrics["epoch"] = epoch
            for name, metric in val_metrics.items():
                history[name] = history.get(name, []) + [metric]
                    
        # 3，early-stopping -------------------------------------------------
        accelerator.wait_for_everyone()
        arr_scores = history[monitor]
        best_score_idx = np.argmax(arr_scores) if mode=="max" else np.argmin(arr_scores)

        if best_score_idx==len(arr_scores)-1:
            unwrapped_net = accelerator.unwrap_model(net)
            accelerator.save(unwrapped_net.state_dict(),ckpt_path)
            accelerator.print(colorful("<<<<<< reach best {0} : {1} >>>>>>".format(monitor,
                 arr_scores[best_score_idx])))

        if len(arr_scores)-best_score_idx>patience:
            accelerator.print(colorful("<<<<<< {} without improvement in {} epoch, early stopping >>>>>>".format(
                monitor,patience)))
            break 
        
    dfhistory = pd.DataFrame(history)
    accelerator.print(dfhistory)
    
    