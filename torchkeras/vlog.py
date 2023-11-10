import os 
import numpy as np 
import pandas as pd 
from torchkeras.pbar import ProgressBar,is_jupyter

class VLog:
    def __init__(self, epochs, monitor_metric='val_loss', monitor_mode='min',  
                 save_path='history.png', figsize = (6,4), bar=True):
        self.figsize = (6,4)
        self.save_path = save_path
        self.bar = bar
        self.metric_name = monitor_metric
        self.metric_mode = monitor_mode
        self.in_jupyter = is_jupyter()
        self.epochs = epochs
        self.batchs = None
        self.history = {}
        self.step, self.epoch = 0, 0
    
    def log_start(self):
        if not self.in_jupyter:
            print('\nView dynamic loss/metric plot: \n'+os.path.abspath(self.save_path))
        dfhistory = pd.DataFrame(self.history)
        x_bounds = [0, min(10,self.epochs)]
        title = f'best {self.metric_name} = ?'
        self.update_graph(title=title, x_bounds = x_bounds)
        if self.bar:
            self.progress = ProgressBar(range(self.epochs))
        
    def log_epoch(self, info):
        self.epoch+=1
        info['epoch'] = self.epoch
        for name, metric in info.items():
            self.history[name] = self.history.get(name, []) + [metric]
        dfhistory = pd.DataFrame(self.history)
        n = len(dfhistory)
        x_bounds = [dfhistory['epoch'].min(), min(10+(n//10)*10,self.epochs)]
        title = self.get_title()
        if self.bar:
            self.progress.update(dfhistory['epoch'].iloc[-1])
            self.step,self.batchs = 0,self.step
        self.update_graph(title = title,x_bounds = x_bounds)
        
        
    def log_step(self, info, training=True):
        if self.bar:
            if training:
                self.step+=1
                if self.batchs:
                    post_log = dict(**{'i':self.step,'n':self.batchs},**info)
                else:
                    post_log = dict(**{'step':self.step},**info)
                self.progress.set_postfix(**post_log)
            else:
                if self.in_jupyter:
                    post_log = info
                    self.progress.set_postfix(**post_log)
                    
    def log_end(self):
        title = self.get_title()
        self.update_graph(title = title)
        dfhistory = pd.DataFrame(self.history)
        if self.bar and self.in_jupyter:
            self.progress.display = True
            self.progress.set_postfix()
            if dfhistory['epoch'].max()<self.epochs:
                self.progress.on_interrupt(msg='early-stopped')
            self.progress.display = False
        return dfhistory 
        
    def get_best_score(self):
        dfhistory = pd.DataFrame(self.history)
        arr_scores = dfhistory[self.metric_name]
        best_score = np.max(arr_scores) if self.metric_mode=="max" else np.min(arr_scores)
        best_epoch = dfhistory.loc[arr_scores==best_score,'epoch'].tolist()[0]
        return (best_epoch, best_score)
        
    def get_title(self):
        best_epoch,best_score = self.get_best_score()
        title = f'best {self.metric_name}={best_score:.4f} (@epoch {best_epoch})'
        return title

    def update_graph(self, title=None, x_bounds=None, y_bounds=None):
        import matplotlib.pyplot as plt
        self.plt = plt
        if not hasattr(self, 'graph_fig'):
            self.graph_fig, self.graph_ax = plt.subplots(1, figsize=self.figsize)
            if self.in_jupyter:
                self.graph_out = display(self.graph_ax.figure, display_id=True)
        self.graph_ax.clear()
        
        dfhistory = pd.DataFrame(self.history)
        epochs = dfhistory['epoch'] if 'epoch' in dfhistory.columns else []
        
        metric_name = self.metric_name.replace('val_','').replace('train_','')
        
        m1 = "train_"+metric_name
        if  m1 in dfhistory.columns:
            train_metrics = dfhistory[m1]
            self.graph_ax.plot(epochs,train_metrics,'bo--',label= m1,clip_on=False)

        m2 = 'val_'+metric_name
        if m2 in dfhistory.columns:
            val_metrics = dfhistory[m2]
            self.graph_ax.plot(epochs,val_metrics,'co-',label =m2,clip_on=False)

        if metric_name in dfhistory.columns:
            metric_values = dfhistory[metric_name]
            self.graph_ax.plot(epochs, metric_values,'co-', label = self.metric_name,clip_on=False)

        self.graph_ax.set_xlabel("epoch")
        self.graph_ax.set_ylabel(metric_name)  
        
        if title:
            self.graph_ax.set_title(title)
                
        if m1 in dfhistory.columns or m2 in dfhistory.columns or self.metric_name in dfhistory.columns:
            self.graph_ax.legend(loc='best')
            
        if len(epochs)>0:
            best_epoch, best_score = self.get_best_score()
            self.graph_ax.plot(best_epoch,best_score,'r*',markersize=15,clip_on=False)

        if x_bounds is not None: self.graph_ax.set_xlim(*x_bounds)
        if y_bounds is not None: self.graph_ax.set_ylim(*y_bounds)
        if self.in_jupyter:
            self.graph_out.update(self.graph_ax.figure)
        self.graph_fig.savefig(self.save_path)
        self.plt.close();
        
if __name__=='__main__':
    import time
    import math,random
    epochs = 10
    batchs = 30
    vlog = VLog(epochs,monitor_metric='val_loss', monitor_mode='min')
    vlog.log_start() 

    for epoch in range(epochs):
        #train
        for step in range(batchs):
            vlog.log_step({'train_loss':100-2.5*epoch+math.sin(2*step/batchs)})
            time.sleep(0.05)
        #eval
        for step in range(20):
            vlog.log_step({'val_loss':100-2*epoch+math.sin(2*step/batchs)},training=False)
            time.sleep(0.05)
        vlog.log_epoch({'val_loss':100 - 2*epoch+2*random.random()-1,
                        'train_loss':100-2.5*epoch+2*random.random()-1})
        if epoch==5:
            break      
    vlog.log_end()