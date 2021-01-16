# -*- coding: utf-8 -*-
import os
import datetime
import numpy as np 
import pandas as pd 
import torch
from collections import OrderedDict
from prettytable import PrettyTable
from torchkeras.summary import summary

__version__ = "2.1.1"

#On macOs, run pytorch and matplotlib at the same time in jupyter should set this.
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE" 

class Model(torch.nn.Module):
    
    # print time bar...
    @staticmethod
    def print_bar(): 
        nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print("\n"+"="*80 + "%s"%nowtime)
        
    def __init__(self,net = None):
        super(Model, self).__init__()
        self.net = net

    def forward(self,x):
        if self.net:
            return self.net.forward(x)
        else:
            raise NotImplementedError
    
    def compile(self, loss_func, 
               optimizer=None, metrics_dict=None,device = None):
        self.loss_func = loss_func
        self.optimizer = optimizer if optimizer else torch.optim.Adam(self.parameters(),lr = 0.001)
        self.metrics_dict = metrics_dict if metrics_dict else {}
        self.history = {}
        self.device = device if torch.cuda.is_available() else None
        if self.device:
            self.to(self.device)
        

    def summary(self,input_shape,input_dtype = torch.FloatTensor, batch_size=-1 ):
        summary(self,input_shape,input_dtype,batch_size)
    
    def train_step(self, features, labels):  
           
        self.train()
        self.optimizer.zero_grad()
        if self.device:
            features = features.to(self.device)
            labels = labels.to(self.device)

        # forward
        predictions = self.forward(features)
        loss = self.loss_func(predictions,labels)
        
        # evaluate metrics
        train_metrics = {"loss":loss.item()}
        for name,metric_func in self.metrics_dict.items():
            train_metrics[name] = metric_func(predictions,labels).item()

        # backward
        loss.backward()

        # update parameters
        self.optimizer.step()
        self.optimizer.zero_grad()
        
        return train_metrics
    
    @torch.no_grad()
    def evaluate_step(self, features,labels):
        
        self.eval()
        
        if self.device:
            features = features.to(self.device)
            labels = labels.to(self.device)
            
        with torch.no_grad():
            predictions = self.forward(features)
            loss = self.loss_func(predictions,labels)
        
        val_metrics = {"val_loss":loss.item()}
        for name,metric_func in self.metrics_dict.items():
            val_metrics["val_"+name] = metric_func(predictions,labels).item()
            
        return val_metrics
        
    
    def fit(self,epochs,dl_train,dl_val = None,log_step_freq = 1):
        
        print("Start Training ...")
        Model.print_bar()
        
        dl_val = dl_val if dl_val else []
        
        for epoch in range(1,epochs+1):
            
            # 1，training loop -------------------------------------------------
            
            train_metrics_sum, step = {}, 0
            for features,labels in dl_train:
                step = step + 1
                train_metrics = self.train_step(features,labels)
    
                for name,metric in train_metrics.items():
                    train_metrics_sum[name] = train_metrics_sum.get(name,0.0)+metric
                    
                if step%log_step_freq == 0:   
                    logs = {"step":step}
                    logs.update({k:round(v/step,3) for k,v in train_metrics_sum.items()})
                    print(logs)
                
            for name,metric_sum in train_metrics_sum.items():
                self.history[name] = self.history.get(name,[])+[metric_sum/step]
                
                
            # 2，validate loop -------------------------------------------------

            val_metrics_sum, step = {}, 0
            for features,labels in dl_val:
                step = step + 1
                val_metrics = self.evaluate_step(features,labels)
                for name,metric in val_metrics.items():
                    val_metrics_sum[name] = val_metrics_sum.get(name,0.0)+metric
            for name,metric_sum in val_metrics_sum.items():
                self.history[name] = self.history.get(name,[])+[metric_sum/step]
                
            # 3，print logs -------------------------------------------------
            
            infos = {"epoch":epoch}
            infos.update({k:round(self.history[k][-1],3) for k in self.history})
            tb = PrettyTable()
            tb.field_names = infos.keys()
            tb.add_row(infos.values())
            print("\n",tb)
            Model.print_bar()
        
        print("Finished Training...")
            
        return pd.DataFrame(self.history)
    
    @torch.no_grad()
    def evaluate(self,dl_val):
        self.eval()
        val_metrics_list = {}
        for features,labels in dl_val:
            val_metrics = self.evaluate_step(features,labels)
            for name,metric in val_metrics.items():
                val_metrics_list[name] = val_metrics_list.get(name,[])+[metric]
        
        return {name:np.mean(metric_list) for name,metric_list in val_metrics_list.items()}
    
    @torch.no_grad()
    def predict(self,dl):
        self.eval()
        if self.device:
            result = torch.cat([self.forward(t[0].to(self.device)) for t in dl])
        else:
            result = torch.cat([self.forward(t[0]) for t in dl])
        return(result.data)
