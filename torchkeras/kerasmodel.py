import sys,datetime
from tqdm import tqdm 
from copy import deepcopy
import numpy as np
import pandas as pd
import torch
from accelerate import Accelerator

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
            
        all_preds = self.accelerator.gather(preds)
        all_labels = self.accelerator.gather(labels)
        all_loss = self.accelerator.gather(loss).sum()
        
        #losses
        step_losses = {self.stage+"_loss":all_loss.item()}
        
        #metrics
        step_metrics = {self.stage+"_"+name:metric_fn(all_preds, all_labels).item() 
                        for name,metric_fn in self.metrics_dict.items()}
        
        if self.optimizer is not None and self.stage=="train":
            step_metrics['lr'] = self.optimizer.state_dict()['param_groups'][0]['lr']
            
        return step_losses,step_metrics

class EpochRunner:
    def __init__(self,steprunner):
        self.steprunner = steprunner
        self.stage = steprunner.stage
        self.steprunner.net.train() if self.stage=="train" else self.steprunner.net.eval()
        self.accelerator = self.steprunner.accelerator
        
    def __call__(self,dataloader):
        loop = tqdm(enumerate(dataloader,start=1), 
                    total =len(dataloader),
                    file=sys.stdout,
                    disable=not self.accelerator.is_local_main_process,
                    ncols = 100
                   )
        
        epoch_losses = {}
        for step, batch in loop: 
            if self.stage=="train":
                step_losses,step_metrics = self.steprunner(batch)
            else:
                with torch.no_grad():
                    step_losses,step_metrics = self.steprunner(batch)
                    
            step_log = dict(step_losses,**step_metrics)
            for k,v in step_losses.items():
                epoch_losses[k] = epoch_losses.get(k,0.0)+v
            
            if step!=len(dataloader):
                loop.set_postfix(**step_log)
            else:
                epoch_metrics = step_metrics
                epoch_metrics.update({self.stage+"_"+name:metric_fn.compute().item() 
                                 for name,metric_fn in self.steprunner.metrics_dict.items()})
                epoch_losses = {k:v/step for k,v in epoch_losses.items()}
                epoch_log = dict(epoch_losses,**epoch_metrics)
                loop.set_postfix(**epoch_log)
                for name,metric_fn in self.steprunner.metrics_dict.items():
                    metric_fn.reset()
        return epoch_log

class KerasModel(torch.nn.Module):
    
    StepRunner,EpochRunner = StepRunner,EpochRunner
    
    def __init__(self,net,loss_fn,metrics_dict=None,optimizer=None,lr_scheduler = None):
        super().__init__()
        self.net,self.loss_fn,self.metrics_dict = net, loss_fn, torch.nn.ModuleDict(metrics_dict) 
        self.optimizer = optimizer if optimizer is not None else torch.optim.Adam(
            self.net.parameters(), lr=1e-3)
        self.lr_scheduler = lr_scheduler

    def forward(self, x):
        return self.net.forward(x)

    def fit(self, train_data, val_data=None, epochs=10,ckpt_path='checkpoint.pt',
            patience=5, monitor="val_loss", mode="min", mixed_precision='no',callbacks = None):
        
        self.__dict__.update(locals())
        self.accelerator = Accelerator(mixed_precision=mixed_precision)
        device = str(self.accelerator.device)
        device_type = '🐌'  if 'cpu' in device else '⚡️'
        self.accelerator.print(colorful("<<<<<< "+device_type +" "+ device +" is used >>>>>>"))
    
        self.net,self.loss_fn,self.metrics_dict,self.optimizer,self.lr_scheduler= self.accelerator.prepare(
            self.net,self.loss_fn,self.metrics_dict,self.optimizer,self.lr_scheduler)
        
        train_dataloader,val_dataloader = self.accelerator.prepare(train_data,val_data)
        
        self.history = {}
        self.callbacks = self.accelerator.prepare(callbacks) if callbacks is not None else []
        
        if self.accelerator.is_local_main_process:
            for callback_obj in self.callbacks:
                callback_obj.on_fit_start(model = self)
        
        for epoch in range(1, epochs+1):

            nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            self.accelerator.print("\n"+"=========="*8 + "%s"%nowtime)
            self.accelerator.print("Epoch {0} / {1}".format(epoch, epochs)+"\n")

            # 1，train -------------------------------------------------  
            train_step_runner = self.StepRunner(
                    net = self.net,
                    loss_fn = self.loss_fn,
                    accelerator = self.accelerator,
                    stage="train",
                    metrics_dict=deepcopy(self.metrics_dict),
                    optimizer = self.optimizer,
                    lr_scheduler = self.lr_scheduler
            )

            train_epoch_runner = self.EpochRunner(train_step_runner)
            train_metrics = {'epoch':epoch}
            train_metrics.update(train_epoch_runner(train_dataloader))
            
            for name, metric in train_metrics.items():
                self.history[name] = self.history.get(name, []) + [metric]
                
            if self.accelerator.is_local_main_process:
                for callback_obj in self.callbacks:
                    callback_obj.on_train_epoch_end(model = self)

            # 2，validate -------------------------------------------------
            if val_dataloader:
                val_step_runner = self.StepRunner(
                    net = self.net,
                    loss_fn = self.loss_fn,
                    accelerator = self.accelerator,
                    stage="val",
                    metrics_dict= deepcopy(self.metrics_dict)
                )
                val_epoch_runner = self.EpochRunner(val_step_runner)
                with torch.no_grad():
                    val_metrics = val_epoch_runner(val_dataloader)

                for name, metric in val_metrics.items():
                    self.history[name] = self.history.get(name, []) + [metric]
                
                if self.accelerator.is_local_main_process:
                    for callback_obj in self.callbacks:
                        callback_obj.on_validation_epoch_end(model = self)

            # 3，early-stopping -------------------------------------------------
            self.accelerator.wait_for_everyone()
            arr_scores = self.history[monitor]
            best_score_idx = np.argmax(arr_scores) if mode=="max" else np.argmin(arr_scores)

            if best_score_idx==len(arr_scores)-1:
                unwrapped_net = self.accelerator.unwrap_model(self.net)
                self.accelerator.save(unwrapped_net.state_dict(),ckpt_path)
                self.accelerator.print(colorful("<<<<<< reach best {0} : {1} >>>>>>".format(monitor,
                     arr_scores[best_score_idx])))

            if len(arr_scores)-best_score_idx>patience:
                self.accelerator.print(colorful(
                    "<<<<<< {} without improvement in {} epoch, early stopping >>>>>>".format(monitor,patience)))
                break 
                
        if self.accelerator.is_local_main_process:
        
            for callback_obj in self.callbacks:
                callback_obj.on_fit_end(model = self)
                
            self.net = self.accelerator.unwrap_model(self.net)
            self.net.load_state_dict(torch.load(ckpt_path))
            dfhistory = pd.DataFrame(self.history)
            self.accelerator.print(dfhistory)
            return dfhistory 
    
    @torch.no_grad()
    def evaluate(self, val_data):
        accelerator = Accelerator()
        self.net,self.loss_fn,self.metrics_dict = accelerator.prepare(self.net,self.loss_fn,self.metrics_dict)
        val_data = accelerator.prepare(val_data)
        val_step_runner = self.StepRunner(net = self.net,stage="val",
                    loss_fn = self.loss_fn,metrics_dict=deepcopy(self.metrics_dict),
                    accelerator = accelerator)
        val_epoch_runner = self.EpochRunner(val_step_runner)
        val_metrics = val_epoch_runner(val_data)
        return val_metrics