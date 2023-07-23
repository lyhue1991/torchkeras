import os 
import sys
import datetime 
from copy import deepcopy
import numpy as np 
import pandas as pd 
from argparse import Namespace 
from torchkeras.utils import is_jupyter

class TensorBoardCallback:
    def __init__(self, save_dir= "runs", model_name="model", 
                 log_weight=False, log_weight_freq=5):
        from torch.utils.tensorboard import SummaryWriter 
        self.__dict__.update(locals())
        nowtime = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        self.log_path = os.path.join(save_dir,model_name,nowtime)
        self.writer = SummaryWriter(self.log_path)
        
    def on_fit_start(self, model:'KerasModel'):
        
        #log weight
        if self.log_weight:
            net = model.accelerator.unwrap_model(model.net)
            for name, param in net.named_parameters():
                self.writer.add_histogram(name, param.clone().cpu().data.numpy(), 0)
            self.writer.flush()
        
    def on_train_epoch_end(self, model:'KerasModel'):
        
        epoch = max(model.history['epoch'])

        #log weight
        net = model.accelerator.unwrap_model(model.net)
        if self.log_weight and epoch%self.log_weight_freq==0:
            for name, param in net.named_parameters():
                self.writer.add_histogram(name, param.clone().cpu().data.numpy(), epoch)
            self.writer.flush()
        
    def on_validation_epoch_end(self, model:"KerasModel"):
        
        dfhistory = pd.DataFrame(model.history)
        n = len(dfhistory)
        epoch = max(model.history['epoch'])
        
        #log metric
        dic = deepcopy(dfhistory.iloc[n-1])
        dic.pop("epoch")
        
        metrics_group = {}
        for key,value in dic.items():
            g = key.replace("train_",'').replace("val_",'')
            metrics_group[g] = dict(metrics_group.get(g,{}), **{key:value})
        for group,metrics in metrics_group.items():
            self.writer.add_scalars(group, metrics, epoch)
        self.writer.flush()
    
    
    def on_fit_end(self, model:"KerasModel"):
        
        #log weight 
        epoch = max(model.history['epoch'])
        if self.log_weight:
            net = model.accelerator.unwrap_model(model.net)
            for name, param in net.named_parameters():
                self.writer.add_histogram(name, param.clone().cpu().data.numpy(), epoch)
            self.writer.flush()
        self.writer.close()
        
        #save history
        dfhistory = pd.DataFrame(model.history)
        dfhistory.to_csv(os.path.join(self.log_path,'dfhistory.csv'),index=None)


class WandbCallback:
    def __init__(self, 
                 project = None,
                 config = None,
                 name = None,
                 save_ckpt = True,
                 save_code = True
                ):
        self.__dict__.update(locals())
        if isinstance(config,Namespace):
            self.config = config.__dict__ 
        if name is None:
            self.name =datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        import wandb 
        self.wb = wandb
        
    def on_fit_start(self, model:'KerasModel'):
        if self.wb.run is None:
            self.wb.init(project=self.project, config = self.config,
                   name = self.name, save_code=self.save_code)
        model.run_id = self.wb.run.id
        
    def on_train_epoch_end(self, model:'KerasModel'):
        pass
    
    def on_validation_epoch_end(self, model:"KerasModel"):
        dfhistory = pd.DataFrame(model.history)
        n = len(dfhistory)
        if n==1:
            for m in dfhistory.columns:
                self.wb.define_metric(name=m, step_metric='epoch', hidden=False if m!='epoch' else True)
            self.wb.define_metric(name='best_'+model.monitor,step_metric='epoch')
        
        dic = dict(dfhistory.iloc[n-1])
        monitor_arr = dfhistory[model.monitor]
        best_monitor_score = monitor_arr.max() if model.mode=='max' else monitor_arr.min()
        dic.update({'best_'+model.monitor:best_monitor_score})
        self.wb.run.summary["best_score"] = best_monitor_score
        self.wb.log(dic)

    def on_fit_end(self, model:"KerasModel"):
        
        #save dfhistory
        dfhistory = pd.DataFrame(model.history)
        dfhistory.to_csv(os.path.join(self.wb.run.dir,'dfhistory.csv'),index=None) 
        
        #save ckpt
        if self.save_ckpt:
            arti_model = self.wb.Artifact('checkpoint', type='model')
            arti_model.add_file(model.ckpt_path)
            self.wb.log_artifact(arti_model)

        run_dir = self.wb.run.dir
        self.wb.finish()

        #local save
        import shutil
        shutil.copy(model.ckpt_path,os.path.join(run_dir,os.path.basename(model.ckpt_path)))
    

class VisProgress:
    def __init__(self):
        pass
        
    def on_fit_start(self,model: 'KerasModel'):
        from .pbar import ProgressBar
        self.progress = ProgressBar(range(model.epochs)) 
        model.EpochRunner.progress = self.progress
        
    def on_train_epoch_end(self,model:'KerasModel'):
        pass
    
    def on_validation_epoch_end(self, model:"KerasModel"):
        dfhistory = pd.DataFrame(model.history)
        self.progress.update(dfhistory['epoch'].iloc[-1])

           
    def on_fit_end(self,  model:"KerasModel"):
        dfhistory = pd.DataFrame(model.history)
        if dfhistory['epoch'].max()<model.epochs:
            self.progress.on_interrupt(msg='earlystopping')
        self.progress.display=False
            
class VisMetric:
    def __init__(self,figsize = (6,4),
                 save_path='history.png'):
        self.figsize = (6,4)
        self.save_path = save_path
        self.in_jupyter = is_jupyter()
        
    def on_fit_start(self,model: 'KerasModel'):
        self.metric =  model.monitor.replace('val_','')
        dfhistory = pd.DataFrame(model.history)
        x_bounds = [0, min(10,model.epochs)]
        title = f'best {model.monitor} = ?'
        self.update_graph(model, title=title, x_bounds = x_bounds)
        
    def on_train_epoch_end(self,model:'KerasModel'):
        pass
    
    def on_validation_epoch_end(self, model:"KerasModel"):
        dfhistory = pd.DataFrame(model.history)
        n = len(dfhistory)
        x_bounds = [dfhistory['epoch'].min(), min(10+(n//10)*10,model.epochs)]
        title = self.get_title(model)
        self.update_graph(model, title = title,x_bounds = x_bounds)
        
            
    def on_fit_end(self,  model:"KerasModel"):
        dfhistory = pd.DataFrame(model.history)
        title = self.get_title(model)
        self.update_graph(model, title = title)
        
    def get_best_score(self, model:'KerasModel'):
        dfhistory = pd.DataFrame(model.history)
        arr_scores = dfhistory[model.monitor]
        best_score = np.max(arr_scores) if model.mode=="max" else np.min(arr_scores)
        best_epoch = dfhistory.loc[arr_scores==best_score,'epoch'].tolist()[0]
        return (best_epoch, best_score)
        
    def get_title(self,  model:'KerasModel'):
        best_epoch,best_score = self.get_best_score(model)
        title = f'best {model.monitor} = {best_score:.4f} (@epoch {best_epoch})'
        return title

    def update_graph(self, model:'KerasModel', title=None, x_bounds=None, y_bounds=None):
        import matplotlib.pyplot as plt
        self.plt = plt
        if not hasattr(self, 'graph_fig'):
            self.graph_fig, self.graph_ax = plt.subplots(1, figsize=self.figsize)
            if self.in_jupyter:
                self.graph_out = display(self.graph_ax.figure, display_id=True)
        self.graph_ax.clear()
        
        dfhistory = pd.DataFrame(model.history)
        epochs = dfhistory['epoch'] if 'epoch' in dfhistory.columns else []
        
        m1 = "train_"+self.metric
        if  m1 in dfhistory.columns:
            train_metrics = dfhistory[m1]
            self.graph_ax.plot(epochs,train_metrics,'bo--',label= m1,clip_on=False)

        m2 = 'val_'+self.metric
        if m2 in dfhistory.columns:
            val_metrics = dfhistory[m2]
            self.graph_ax.plot(epochs,val_metrics,'co-',label =m2,clip_on=False)

        if self.metric in dfhistory.columns:
            metric_values = dfhistory[self.metric]
            self.graph_ax.plot(epochs, metric_values,'co-', label = self.metric,clip_on=False)

        self.graph_ax.set_xlabel("epoch")
        self.graph_ax.set_ylabel(self.metric)  
        if title:
             self.graph_ax.set_title(title)
        if m1 in dfhistory.columns or m2 in dfhistory.columns or self.metric in dfhistory.columns:
            self.graph_ax.legend(loc='best')
            
        if len(epochs)>0:
            best_epoch, best_score = self.get_best_score(model)
            self.graph_ax.plot(best_epoch,best_score,'r*',markersize=15,clip_on=False)

        if x_bounds is not None: self.graph_ax.set_xlim(*x_bounds)
        if y_bounds is not None: self.graph_ax.set_ylim(*y_bounds)
        if self.in_jupyter:
            self.graph_out.update(self.graph_ax.figure)
        else:
            self.graph_fig.savefig(self.save_path)
        self.plt.close();
        

class VisDisplay:
    def __init__(self,display_fn,model=None,init_display=True,dis_period=1):
        from ipywidgets import Output 
        self.display_fn = display_fn
        self.init_display = init_display
        self.dis_period = dis_period
        self.out = Output()
        
        if self.init_display:
            display(self.out)
            with self.out:
                self.display_fn(model)
        
    def on_fit_start(self,model: 'KerasModel'):
        if not self.init_display:
            display(self.out)

    def on_train_epoch_end(self,model:'KerasModel'):
        pass
    
    def on_validation_epoch_end(self, model:"KerasModel"):
        if len(model.history['epoch'])%self.dis_period==0:
            self.out.clear_output()
            with self.out:
                self.display_fn(model)
           
    def on_fit_end(self,  model:"KerasModel"):
        pass
        
        
class EpochCheckpoint:
    def __init__(self, ckpt_dir= "weights", 
                 save_freq=1, max_ckpt=10):
        self.__dict__.update(locals())
        self.ckpt_idx=0
        
    def on_fit_start(self, model:'KerasModel'):
        if not os.path.exists(self.ckpt_dir):
            os.mkdir(self.ckpt_dir)
        self.ckpt_list = ['' for i in range(self.max_ckpt)]
        
    def on_train_epoch_end(self, model:'KerasModel'):
        pass
        
    def on_validation_epoch_end(self, model:"KerasModel"):
        dfhistory = pd.DataFrame(model.history)
        epoch = dfhistory['epoch'].iloc[-1]
        if epoch>0 and epoch%self.save_freq==0:
            ckpt_path = os.path.join(self.ckpt_dir,f'checkpoint_epoch{epoch}.pt')
            net_dict = model.accelerator.get_state_dict(model.net)
            model.accelerator.save(net_dict,ckpt_path)
            
            if self.ckpt_list[self.ckpt_idx]!='':
                os.remove(self.ckpt_list[self.ckpt_idx])
            self.ckpt_list[self.ckpt_idx] = ckpt_path 
            self.ckpt_idx = (self.ckpt_idx+1)%self.max_ckpt

    def on_fit_end(self, model:"KerasModel"):
        pass