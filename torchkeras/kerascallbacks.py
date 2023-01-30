import os 
import datetime
from copy import deepcopy
import numpy as np 
import pandas as pd 
import wandb
import plotly.graph_objs as go
from torch.utils.tensorboard import SummaryWriter 

def plot_metric(dfhistory, metric):
    train_metrics = dfhistory["train_"+metric].values.tolist()
    val_metrics = dfhistory['val_'+metric].values.tolist()
    epochs = list(range(1, len(train_metrics) + 1))
    
    train_scatter = go.Scatter(x = epochs, y=train_metrics, mode = "lines+markers",
                               name = 'train_'+metric,marker = dict(size=8,color="blue"),
                                line= dict(width=2,color="blue",dash="dash"))
    val_scatter = go.Scatter(x = epochs, y=val_metrics, mode = "lines+markers",
                            name = 'val_'+metric,marker = dict(size=10,color="red"),
                            line= dict(width=2,color="red",dash="solid"))
    fig = go.Figure(data = [train_scatter,val_scatter])
    
    return fig

def getNotebookPath():
    from jupyter_server import serverapp
    from jupyter_server.utils import url_path_join
    from pathlib import Path
    import requests,re
    kernelIdRegex = re.compile(r"(?<=kernel-)[\w\d\-]+(?=\.json)")
    kernelId = kernelIdRegex.search(get_ipython().config["IPKernelApp"]["connection_file"])[0]
    for jupServ in serverapp.list_running_servers():
        for session in requests.get(url_path_join(jupServ["url"], "api/sessions"),
                                    params={"token":jupServ["token"]}).json():
            if kernelId == session["kernel"]["id"]:
                return str(Path(jupServ["root_dir"]) / session["notebook"]['path']) 
    raise Exception('failed to get current notebook path')
    
    
class TensorBoardCallback:
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
        
        
class WandbCallback:
    def __init__(self, 
                 project="model",
                 config = None,
                 name = None,
                 save_code = True,
                 save_ckpt = True
                ):
        self.project = project
        self.config = config if config else {}
        self.name = name if name is not None else datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        self.save_code = save_code
        self.save_ckpt = save_ckpt
        
    def on_fit_start(self, model:'KerasModel'):
        run = wandb.init(project=self.project, config = self.config,
                   name = self.name)
        model.run_id = run.id
        

    def on_train_epoch_end(self, model:'KerasModel'):
        pass
    
    def on_validation_epoch_end(self, model:"KerasModel"):
        dfhistory = pd.DataFrame(model.history)
        epoch = max(dfhistory['epoch'])
        
        if epoch==1:
            for m in dfhistory.columns:
                wandb.define_metric(name=m, step_metric='epoch', hidden=False if m!='epoch' else True)
            wandb.define_metric(name='best_'+model.monitor,step_metric='epoch')
        
        dic = dict(dfhistory.loc[epoch-1])
        monitor_arr = dfhistory[model.monitor]
        best_monitor_score = monitor_arr.max() if model.mode=='max' else monitor_arr.min()
        dic.update({'best_'+model.monitor:best_monitor_score})
        wandb.run.summary["best_score"] = best_monitor_score
        wandb.log(dic)

    def on_fit_end(self, model:"KerasModel"):
        
        #save dfhistory
        dfhistory = pd.DataFrame(model.history)
        dfhistory.to_csv(os.path.join(wandb.run.dir,'dfhistory.csv'),index=None) 
        
        #save ckpt
        if self.save_ckpt:
            wandb.save(model.ckpt_path)
        
        #save code
        if self.save_code:
            try:
                current_file = eval('__file__')
                wandb.save(current_file)
            except Exception as err1:
                try:
                    current_file = getNotebookPath()
                    wandb.save(current_file)
                except Exception as err2:
                    print(err1,err2)
 
                
        #plotly metrics
        metrics = [x.replace('train_','').replace('val_','') for x in dfhistory.columns if 'train_' in x] 
        metric_fig = {m+'_curve':plot_metric(dfhistory,m) for m in metrics}
        wandb.log(metric_fig)
        wandb.finish()