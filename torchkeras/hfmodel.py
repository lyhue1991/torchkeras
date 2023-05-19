import torch
from transformers import Trainer,TrainingArguments,EarlyStoppingCallback,TrainerCallback
import numpy as np
import pandas as pd 
import os 
import matplotlib.pyplot as plt    
        
class HfModel(torch.nn.Module):
    def __init__(self, net, loss_fn, metrics_dict=None):
        super().__init__()
        self.net,self.loss_fn = net,loss_fn
        self.metrics_dict = metrics_dict if metrics_dict is not None else {}
        self.input_name = self.net.forward.__code__.co_varnames[1]
        self.label_names = ['labels']
        self.trainer = None
        
    def forward(self,**args):
        inputs = {}
        for k,v in args.items():
            if k in self.net.forward.__code__.co_varnames:
                inputs[k]=v
                
        logits = self.net.forward(**inputs)
        out = {'logits':logits,'loss':torch.tensor(0.0)}
        if 'labels' in args.keys():  
            labels = args['labels']
            out['loss'] = self.loss_fn(logits,labels)
        return out 
            
        
    def compute_metrics(self, eval_preds):
        logits,labels = eval_preds
        for k,m in self.metrics_dict.items():
            m(torch.from_numpy(logits),torch.from_numpy(labels))
        result = {k:m.compute() for k,m in self.metrics_dict.items()}
        for k,m in self.metrics_dict.items():
            m.reset()
        return result 
    
    def fit(self, train_data, val_data=None, epochs=10, 
        output_dir='output_dir',
        patience=5, monitor="eval_loss", mode="min", 
        plot=False, wandb=False,logging_steps=20,
        gradient_accumulation_steps = 1):
        
        if wandb ==False:
            os.environ["WANDB_DISABLED"] = "true"
        else:
            os.environ["WANDB_DISABLED"] = "false"
        
        training_args = TrainingArguments(
            output_dir = output_dir,
            num_train_epochs = epochs,
            logging_steps = logging_steps,
            gradient_accumulation_steps = gradient_accumulation_steps,
            evaluation_strategy="steps", #epoch
            metric_for_best_model= monitor,
            greater_is_better= False if mode=='min' else True,
            report_to='wandb' if wandb else None,
            load_best_model_at_end=True,
            label_names = self.label_names
        )
        
        def collate_fn(examples):
            batch = train_data.collate_fn(examples)
            if isinstance(batch,dict):
                return batch
            elif isinstance(batch, (list,tuple)):
                if len(batch)==2:
                    return {self.input_name:batch[0],'labels':batch[1]}
            else:
                raise Exception('dataset format error!')
                
        
        callbacks = [EarlyStoppingCallback(early_stopping_patience= patience)]
        
        if plot:
            callbacks.append(VisCallback())
        
            
        self.trainer = Trainer(
            self,
            training_args,
            train_dataset=train_data.dataset,
            eval_dataset=val_data.dataset,
            compute_metrics=self.compute_metrics,
            callbacks = callbacks,
            data_collator=collate_fn
        )
        
        self.trainer.train()
        
    def evaluate(self,val_data):
        return self.trainer.evaluate(val_data.dataset)



class VisCallback(TrainerCallback):
    def __init__(self, figsize=(6,4), update_freq=1):
        self.figsize = figsize
        self.update_freq = update_freq

    def on_train_begin(self, args, state, control, **kwargs):
        metric = args.metric_for_best_model
        self.metric = metric
        dfhistory = pd.DataFrame()
        x_bounds = [0, 100]
        title = f'best {metric} = ?'
        self.update_graph(dfhistory, self.metric.replace('eval_',''), 
                             x_bounds = x_bounds, 
                             title=title, 
                             figsize = self.figsize)
        
    def on_evaluate(self, args, state, control, metrics, **kwargs):
        
        dfhistory = self.get_history(state)
        n = dfhistory['step'].max()
        if n%self.update_freq==0:
            x_bounds = [dfhistory['step'].min(), 200+(n//200)*200]
            arr_scores = dfhistory[self.metric]
            best_score = np.max(arr_scores) if args.greater_is_better==True else np.min(arr_scores)

            title = f'best {self.metric} = {best_score:.4f}'
            self.update_graph(dfhistory, self.metric.replace('eval_',''), x_bounds = x_bounds, 
                                 title = title, figsize = self.figsize)
            
    def on_train_end(self, args, state, control, **kwargs):
        dfhistory = self.get_history(state)
        arr_scores = dfhistory[self.metric]
        best_score = np.max(arr_scores) if args.greater_is_better==True else np.min(arr_scores)
        title = f'best {self.metric} = {best_score:.4f}'
        self.update_graph(dfhistory, self.metric.replace('eval_',''), 
                             title = title, figsize = self.figsize)
        plt.close()
        
    def get_history(self,state):
        log_history = state.log_history  
        train_history = [x for x in log_history if 'loss' in x.keys()]
        eval_history = [x for x in log_history if 'eval_loss'  in x.keys()]

        dfhistory_train = pd.DataFrame(train_history)
        dfhistory_eval = pd.DataFrame(eval_history)  
        dfhistory = dfhistory_train.merge(dfhistory_eval,on=['step','epoch'])
        return dfhistory
        
    def update_graph(self, dfhistory, metric, x_bounds=None, 
                     y_bounds=None, title = None, figsize=(6,4)):
        
        from IPython.display import display
        if not hasattr(self, 'graph_fig'):
            self.graph_fig, self.graph_ax = plt.subplots(1, figsize=figsize)
            self.graph_out = display(self.graph_ax.figure, display_id=True)

        self.graph_ax.clear()
        steps = dfhistory['step'] if 'step' in dfhistory.columns else []

        m1 = metric
        if  m1 in dfhistory.columns:
            train_metrics = dfhistory[m1]
            self.graph_ax.plot(steps,train_metrics,'bo--',label= m1)

        m2 = 'eval_'+metric
        if m2 in dfhistory.columns:
            val_metrics = dfhistory[m2]
            self.graph_ax.plot(steps,val_metrics,'ro-',label =m2)

        self.graph_ax.set_xlabel("step")
        self.graph_ax.set_ylabel(metric)  

        if title:
             self.graph_ax.set_title(title)

        if m1 in dfhistory.columns or m2 in dfhistory.columns or metric in dfhistory.columns:
            self.graph_ax.legend(loc='best')

        if x_bounds is not None: self.graph_ax.set_xlim(*x_bounds)
        if y_bounds is not None: self.graph_ax.set_ylim(*y_bounds)
        self.graph_out.update(self.graph_ax.figure);