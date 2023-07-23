import sys,datetime
from tqdm import tqdm
from copy import deepcopy
import numpy as np
import pandas as pd
import torch
from accelerate import Accelerator

class StepRunner:
    def __init__(self, net, loss_fn, accelerator=None, stage = "train", metrics_dict = None, 
                 optimizer = None, lr_scheduler = None
                 ):
        self.net,self.loss_fn,self.metrics_dict,self.stage = net,loss_fn,metrics_dict,stage
        self.optimizer,self.lr_scheduler = optimizer,lr_scheduler
        self.accelerator = accelerator
        if self.stage=='train':
            self.net.train() 
        else:
            self.net.eval()
    
    def __call__(self, batch):
        features,labels = batch 
        
        #loss
        with self.accelerator.autocast():
            preds = self.net(features)
            loss = self.loss_fn(preds,labels)

        #backward()
        if self.stage=="train" and self.optimizer is not None:
            self.accelerator.backward(loss)
            if self.accelerator.sync_gradients:
                self.accelerator.clip_grad_norm_(self.net.parameters(), 1.0)
            self.optimizer.step()
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
            self.optimizer.zero_grad()
            
        all_loss = self.accelerator.gather(loss).sum()
        all_preds = self.accelerator.gather(preds)
        all_labels = self.accelerator.gather(labels)
        
        #losses (or plain metrics that can be averaged)
        step_losses = {self.stage+"_loss":all_loss.item()}
        
        #metrics (stateful metrics)
        step_metrics = {self.stage+"_"+name:metric_fn(all_preds, all_labels).item() 
                        for name,metric_fn in self.metrics_dict.items()}
        
        if self.stage=="train":
            if self.optimizer is not None:
                step_metrics['lr'] = self.optimizer.state_dict()['param_groups'][0]['lr']
            else:
                step_metrics['lr'] = 0.0
        return step_losses,step_metrics

class EpochRunner:
    def __init__(self,steprunner,quiet=False):
        self.steprunner = steprunner
        self.stage = steprunner.stage
        self.accelerator = steprunner.accelerator
        self.net = steprunner.net
        self.quiet = quiet
        
    def __call__(self,dataloader):
        n = dataloader.size  if hasattr(dataloader,'size') else len(dataloader)
        loop = tqdm(enumerate(dataloader,start=1), 
                    total=n,
                    file=sys.stdout,
                    disable=not self.accelerator.is_local_main_process or self.quiet,
                    ncols=100
                   )
        epoch_losses = {}
        
        for step, batch in loop: 
            with self.accelerator.accumulate(self.net):
                step_losses,step_metrics = self.steprunner(batch)   
                step_log = dict(step_losses,**step_metrics)
                for k,v in step_losses.items():
                    epoch_losses[k] = epoch_losses.get(k,0.0)+v
                    
                if step<n:
                    loop.set_postfix(**step_log)
                    
                    if hasattr(self,'progress') and self.accelerator.is_local_main_process:
                        post_log = dict(**{'i':step,'n':n},**step_log)
                        self.progress.set_postfix(**post_log)

                elif step==n:
                    epoch_metrics = step_metrics
                    epoch_metrics.update({self.stage+"_"+name:metric_fn.compute().item() 
                                     for name,metric_fn in self.steprunner.metrics_dict.items()})
                    epoch_losses = {k:v/step for k,v in epoch_losses.items()}
                    epoch_log = dict(epoch_losses,**epoch_metrics)
                    loop.set_postfix(**epoch_log)
            
                    
                    if hasattr(self,'progress') and self.accelerator.is_local_main_process:
                        post_log = dict(**{'i':step,'n':n},**epoch_log)
                        self.progress.set_postfix(**post_log)
                    
                    for name,metric_fn in self.steprunner.metrics_dict.items():
                        metric_fn.reset()  
                else:
                    break
        return epoch_log

class KerasModel(torch.nn.Module):
    
    StepRunner,EpochRunner = StepRunner,EpochRunner
    
    def __init__(self,net,loss_fn,metrics_dict=None,optimizer=None,lr_scheduler = None):
        super().__init__()
        self.net,self.loss_fn,self.metrics_dict = net, loss_fn, torch.nn.ModuleDict(metrics_dict) 
        self.optimizer = optimizer if optimizer is not None else torch.optim.Adam(
            self.net.parameters(), lr=3e-4)
        self.lr_scheduler = lr_scheduler
        self.from_scratch = True
        
    def save_ckpt(self, ckpt_path='checkpoint', accelerator= None):
        accelerator = accelerator if accelerator is not None else self.accelerator
        net_dict = accelerator.get_state_dict(self.net)
        accelerator.save(net_dict,ckpt_path)
      
    def load_ckpt(self, ckpt_path='checkpoint'):
        self.net.load_state_dict(torch.load(ckpt_path,map_location='cpu'))
        self.from_scratch = False

    def forward(self, x):
        return self.net.forward(x)
    
    def fit(self, train_data, val_data=None, epochs=10, ckpt_path='checkpoint',
            patience=5, monitor="val_loss", mode="min", callbacks=None, 
            plot=True,  wandb=False, quiet=None, 
            mixed_precision='no', cpu=False, gradient_accumulation_steps=1):
        from torchkeras.utils import colorful,is_jupyter
        
        self.__dict__.update(locals())
        self.accelerator = Accelerator(mixed_precision=mixed_precision,cpu=cpu,
            gradient_accumulation_steps=gradient_accumulation_steps)
        device = str(self.accelerator.device)
        device_type = '🐌'  if 'cpu' in device else ('⚡️' if 'cuda' in device else '🚀')
        self.accelerator.print(
            colorful("<<<<<< "+device_type +" "+ device +" is used >>>>>>"))
    
        self.net,self.loss_fn,self.metrics_dict,self.optimizer,self.lr_scheduler= self.accelerator.prepare(
            self.net,self.loss_fn,self.metrics_dict,self.optimizer,self.lr_scheduler)
        
        train_dataloader,val_dataloader = self.accelerator.prepare(train_data,val_data)
        train_dataloader.size = train_data.size if hasattr(train_data,'size') else len(train_data)
        if val_data:
            val_dataloader.size = val_data.size if hasattr(val_data,'size') else len(val_data)
        
        
        self.history = {}
        callbacks = callbacks if callbacks is not None else []
        
        if bool(plot):
            if is_jupyter():
                from torchkeras.kerascallbacks import VisProgress
                callbacks = [VisProgress()]+callbacks
            from torchkeras.kerascallbacks import VisMetric
            callbacks = [VisMetric()]+callbacks
            
        if wandb!=False:
            from torchkeras.kerascallbacks import WandbCallback
            project = wandb if isinstance(wandb,str) else 'torchkeras'
            callbacks.append(WandbCallback(project=project))
            
        self.callbacks = [self.accelerator.prepare(x) for x in callbacks]
        
        if self.accelerator.is_local_main_process:
            [cb.on_fit_start(model = self) for cb in self.callbacks if hasattr(cb,'on_fit_start')]
                
        start_epoch = 1 if self.from_scratch else 0
        
        if quiet is None:
            if is_jupyter():
                quiet = True
            else:
                quiet = False
        
        quiet_fn = (lambda epoch:quiet) if isinstance(quiet,bool) else (
            (lambda epoch:epoch>quiet) if isinstance(quiet,int) else quiet)
        
            
        for epoch in range(start_epoch,epochs+1):
            should_quiet = quiet_fn(epoch)
        
            if not should_quiet:
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
                    optimizer = self.optimizer if epoch>0 else None,
                    lr_scheduler = self.lr_scheduler if epoch>0 else None
            )

            train_epoch_runner = self.EpochRunner(train_step_runner,should_quiet)
            train_metrics = {'epoch':epoch}
            train_metrics.update(train_epoch_runner(train_dataloader))

            for name, metric in train_metrics.items():
                self.history[name] = self.history.get(name, []) + [metric]

            if self.accelerator.is_local_main_process:
                [cb.on_train_epoch_end(model = self) for cb in self.callbacks 
                 if hasattr(cb,'on_train_epoch_end')]
                
            # 2，validate -------------------------------------------------
            if val_dataloader is not None:
                val_step_runner = self.StepRunner(
                    net = self.net,
                    loss_fn = self.loss_fn,
                    accelerator = self.accelerator,
                    stage="val",
                    metrics_dict= deepcopy(self.metrics_dict)
                )
                val_epoch_runner = self.EpochRunner(val_step_runner,should_quiet)
                with torch.no_grad():
                    val_metrics = val_epoch_runner(val_dataloader)

                for name, metric in val_metrics.items():
                    self.history[name] = self.history.get(name, []) + [metric]
                
            if self.accelerator.is_local_main_process:
                [cb.on_validation_epoch_end(model = self) for cb in self.callbacks 
                 if hasattr(cb,'on_validation_epoch_end')]

            # 3，early-stopping -------------------------------------------------
            self.accelerator.wait_for_everyone()
            arr_scores = self.history[monitor]
            best_score_idx = np.argmax(arr_scores) if mode=="max" else np.argmin(arr_scores)

            if best_score_idx==len(arr_scores)-1 and self.accelerator.is_local_main_process:
                self.save_ckpt(ckpt_path,accelerator = self.accelerator)
                if not should_quiet:
                    self.accelerator.print(colorful("<<<<<< reach best {0} : {1} >>>>>>".format(
                        monitor,arr_scores[best_score_idx])))

            if len(arr_scores)-best_score_idx>patience:
                self.accelerator.print(colorful(
                    "<<<<<< {} without improvement in {} epoch,""early stopping >>>>>>"
                ).format(monitor,patience))
                break; 
                
        if self.accelerator.is_local_main_process:   
            dfhistory = pd.DataFrame(self.history)
            [cb.on_fit_end(model = self) for cb in self.callbacks 
                 if hasattr(cb,'on_fit_end')]
            self.net = self.accelerator.unwrap_model(self.net)
            self.net.cpu()
            self.load_ckpt(ckpt_path)
            torch.save(dfhistory,'dfhistory.pt')
            return dfhistory
        
    def evaluate(self, val_data, quiet=False):
        accelerator = Accelerator() if not hasattr(self,'accelerator') else self.accelerator
        self.net,self.loss_fn,self.metrics_dict = accelerator.prepare(
            self.net,self.loss_fn,self.metrics_dict)
        val_data = accelerator.prepare(val_data)
        val_step_runner = self.StepRunner(net = self.net,stage="val",
                    loss_fn = self.loss_fn,metrics_dict=deepcopy(self.metrics_dict),
                    accelerator = accelerator)
        val_epoch_runner = self.EpochRunner(val_step_runner,quiet=quiet)
        with torch.no_grad():
            val_metrics = val_epoch_runner(val_data)
        if accelerator.is_local_main_process:
            torch.save(val_metrics,'val_metrics.pt')
        return val_metrics
    
    def fit_ddp(self,num_processes,train_data,
            val_data=None, epochs=10, ckpt_path='checkpoint',
            patience=5, monitor="val_loss", mode="min", callbacks=None, 
            plot=True, wandb=False, quiet=None, 
            mixed_precision='no', cpu=False, gradient_accumulation_steps=1
           ):
        from accelerate import notebook_launcher
        train_size = train_data.size if hasattr(train_data,'size') else len(train_data)
        train_data.size = train_size//num_processes
        if val_data:
            val_size = val_data.size if hasattr(val_data,'size') else len(val_data)
            val_data.size = val_size//num_processes
            
        args = (train_data,val_data,epochs,ckpt_path,patience,monitor,mode,
            callbacks,plot,wandb,quiet,mixed_precision,cpu,gradient_accumulation_steps)
        notebook_launcher(self.fit, args, num_processes=num_processes)
        dfhistory = torch.load('dfhistory.pt')
        
        train_data.size = train_size 
        if val_data:
            val_data.size = val_size 
        return dfhistory
    
    def evaluate_ddp(self, num_processes, val_data, quiet=False):
        from accelerate import notebook_launcher
        val_size = val_data.size if hasattr(val_data,'size') else len(val_data)
        val_data.size = val_size//num_processes
        args = (val_data,quiet)
        notebook_launcher(self.evaluate, args, num_processes=num_processes)
        val_metrics = torch.load('val_metrics.pt')
        val_data.size = val_size
        return val_metrics