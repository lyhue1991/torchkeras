from torch.utils.tensorboard import SummaryWriter 
import os 
import datetime
from copy import deepcopy
import numpy as np 
import pandas as pd 
import wandb

class BaseCallback:
    def __init__(self):
        pass 
    
    def on_fit_start(self, model:'KerasModel'):
        pass
    
    def on_train_epoch_end(self, model:'KerasModel'):
        pass
    
    def on_validation_epoch_end(self, model:'KerasModel'):
        pass
    
    def on_fit_end(self, model:"KerasModel"):
        pass
     
class TensorBoardCallback(BaseCallback):
    def __init__(self, save_dir= "runs", model_name="model", 
                 log_weight=False, log_weight_freq=5):
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
        epoch = max(model.history['epoch'])
        
        #log metric
        dic = deepcopy(dfhistory.loc[epoch-1])
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
        
        
class WandbCallback(BaseCallback):
    def __init__(self, 
                 project="model",
                 config = None
                ):
        self.project = project
        self.config = config if config else {}
        
    def on_fit_start(self, model:'KerasModel'):
        #wandb.login(key="xxx")
        wandb.init(project=self.project,config = self.config)
                 
    def on_train_epoch_end(self, model:'KerasModel'):
        pass
    
    def on_validation_epoch_end(self, model:"KerasModel"):
        
        #log metric
        dfhistory = pd.DataFrame(model.history)
        epoch = max(dfhistory['epoch'])
        dic = deepcopy(dict(dfhistory.loc[epoch-1]))
        wandb.log(dic)

    def on_fit_end(self, model:"KerasModel"):
        #save history
        dfhistory = pd.DataFrame(model.history)
        dfhistory.to_csv(os.path.join(wandb.run.dir,'dfhistory.csv'),index=None) 
        wandb.finish()