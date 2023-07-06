import pandas as pd 
import numpy as np 
from IPython.display  import display 

#Visual Metrics in Notebook for Catboost
class VisCallback:
    def __init__(self, figsize = (8,6), metric='val_Accuracy', mode='max'):
        import matplotlib.pyplot as plt
        self.plt = plt
        self.figsize = figsize
        self.metric = metric
        self.mode = mode
        x_bounds = [0, 10]
        title = f'best {metric} = ?'
        self.update_graph(None, title=title, x_bounds = x_bounds)
        
    def get_dfhistory(self, info):
        from copy import deepcopy
        if not hasattr(info,'metrics'):
            return pd.DataFrame() 
        
        dic = deepcopy(info.metrics) 
        if 'learn' in dic:
            dic['train'] = dic['learn']
            dic.pop('learn')

        if 'validation' in dic:
            dic['val'] = dic['validation']
            dic.pop('validation')

        dfhis_train = pd.DataFrame(dic['train']) 
        dfhis_train.columns = ['train_'+x for x in dfhis_train.columns]

        dfhis_val = pd.DataFrame(dic['val']) 
        dfhis_val.columns = ['val_'+x for x in dfhis_val.columns]
        dfhistory = dfhis_train.join(dfhis_val)
        dfhistory['iteration'] = range(1,len(dfhistory)+1)
        
        return dfhistory 

    def after_iteration(self, info):
        dfhistory = self.get_dfhistory(info)
        n = len(dfhistory)
        x_bounds = [dfhistory['iteration'].min(), 10+(n//10)*10]
        title = self.get_title(info)
        self.update_graph(info, title = title, x_bounds = x_bounds)
        return True
        
    def get_best_score(self, info):
        dfhistory = self.get_dfhistory(info)
        arr_scores = dfhistory[self.metric]
        best_score = np.max(arr_scores) if self.mode=="max" else np.min(arr_scores)
        best_iteration = dfhistory.loc[arr_scores==best_score,'iteration'].tolist()[0]
        return (best_iteration, best_score)
        
    def get_title(self,  info):
        best_iteration,best_score = self.get_best_score(info)
        title = f'best {self.metric} = {best_score:.4f} (@iteration {best_iteration})'
        return title

    def update_graph(self, info, title=None, x_bounds=None, y_bounds=None):
        if not hasattr(self, 'graph_fig'):
            self.graph_fig, self.graph_ax = self.plt.subplots(1, figsize=self.figsize)
            self.graph_out = display(self.graph_ax.figure, display_id=True)
        self.graph_ax.clear()
        dfhistory = self.get_dfhistory(info)
        iterations = dfhistory['iteration'] if 'iteration' in dfhistory.columns else []
        
        m1 = "train_"+self.metric.replace('val_','')
        if  m1 in dfhistory.columns:
            train_metrics = dfhistory[m1]
            
            self.graph_ax.plot(iterations,train_metrics,'bo--',label= m1)

        m2 = 'val_'+self.metric.replace('val_','')
        if m2 in dfhistory.columns:
            val_metrics = dfhistory[m2]
            self.graph_ax.plot(iterations,val_metrics,'co-',label =m2)


        self.graph_ax.set_xlabel("iteration")
        self.graph_ax.set_ylabel(self.metric.replace('val_',''))  
        if title:
             self.graph_ax.set_title(title)
        if m1 in dfhistory.columns or m2 in dfhistory.columns or self.metric in dfhistory.columns:
            self.graph_ax.legend(loc='best')
            
        if len(iterations)>0:
            best_iteration, best_score = self.get_best_score(info)
            self.graph_ax.plot(best_iteration,best_score,'r*',markersize=15)

        if x_bounds is not None: self.graph_ax.set_xlim(*x_bounds)
        if y_bounds is not None: self.graph_ax.set_ylim(*y_bounds)
        self.graph_out.update(self.graph_ax.figure)
        self.plt.close();
        