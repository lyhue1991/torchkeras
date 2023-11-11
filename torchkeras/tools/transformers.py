import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from transformers import TrainerCallback
from torchkeras.pbar import is_jupyter

class VLogCallback(TrainerCallback):
    def __init__(self, figsize=(6,4), update_freq=1, save_path='history.png'):
        self.figsize = figsize
        self.update_freq = update_freq
        self.save_path = save_path
        self.in_jupyter = is_jupyter()

    def on_train_begin(self, args, state, control, **kwargs):
        metric = args.metric_for_best_model
        self.greater_is_better = args.greater_is_better 
        self.prefix = 'val_' if metric.startswith('val_') else 'eval_'
        self.metric =  metric
        
        dfhistory = pd.DataFrame()
        x_bounds = [0, args.logging_steps*10]
        self.update_graph(dfhistory, self.metric.replace(self.prefix,''), 
                             x_bounds = x_bounds, 
                             figsize = self.figsize)
        
    def on_evaluate(self, args, state, control, metrics, **kwargs):
        dfhistory = self.get_history(state)
        n = dfhistory['step'].max()
        if n%self.update_freq==0:
            x_bounds = [dfhistory['step'].min(), 
                        10*args.logging_steps+(n//(args.logging_steps*10))*args.logging_steps*10]
            self.update_graph(dfhistory, self.metric.replace(self.prefix,''), 
                              x_bounds = x_bounds, figsize = self.figsize)
            
    def on_train_end(self, args, state, control, **kwargs):
        dfhistory = self.get_history(state)
        self.update_graph(dfhistory, self.metric.replace(self.prefix,''), 
                             figsize = self.figsize)
        
    def get_history(self,state):
        log_history = state.log_history  
        train_history = [x for x in log_history if 'loss' in x.keys()]
        eval_history = [x for x in log_history if 'eval_loss'  in x.keys()]

        dfhistory_train = pd.DataFrame(train_history)
        dfhistory_eval = pd.DataFrame(eval_history)  
        dfhistory = dfhistory_train.merge(dfhistory_eval,on=['step','epoch'])
        if self.prefix=='val_':
            dfhistory.columns = [x.replace('eval_','val_') for x in dfhistory.columns]
        return dfhistory
    
    def get_best_score(self, dfhistory):
        arr_scores = dfhistory[self.metric]
        best_score = np.max(arr_scores) if self.greater_is_better==True else np.min(arr_scores)
        best_step = dfhistory.loc[arr_scores==best_score,'step'].tolist()[0]
        return (best_step, best_score)

    def update_graph(self, dfhistory, metric, x_bounds=None, 
                     y_bounds=None, figsize=(6,4)):
        
        from IPython.display import display
        if not hasattr(self, 'graph_fig'):
            self.graph_fig, self.graph_ax = plt.subplots(1, figsize=figsize)
            self.graph_out = display(self.graph_ax.figure, display_id=True)

        self.graph_ax.clear()
        steps = dfhistory['step'] if 'step' in dfhistory.columns else []

        m1 = metric
        if  m1 in dfhistory.columns:
            train_metrics = dfhistory[m1]
            self.graph_ax.plot(steps,train_metrics,'bo--',label= m1,clip_on=False)

        m2 = self.prefix+metric
        if m2 in dfhistory.columns:
            val_metrics = dfhistory[m2]
            self.graph_ax.plot(steps,val_metrics,'co-',label =m2,clip_on=False)

        self.graph_ax.set_xlabel("step")
        self.graph_ax.set_ylabel(metric)  

        if m1 in dfhistory.columns or m2 in dfhistory.columns or metric in dfhistory.columns:
            self.graph_ax.legend(loc='best')
            
        if len(steps)>0:
            best_step, best_score = self.get_best_score(dfhistory)
            self.graph_ax.plot(best_step,best_score,'r*',markersize=15,clip_on=False)
            title = f'best {self.metric} = {best_score:.4f} (@step {best_step})'
            self.graph_ax.set_title(title)
        else:
            title = f'best {self.metric} = ?'
            self.graph_ax.set_title(title)
            
        if x_bounds is not None: self.graph_ax.set_xlim(*x_bounds)
        if y_bounds is not None: self.graph_ax.set_ylim(*y_bounds)
        if self.in_jupyter:
            self.graph_out.update(self.graph_ax.figure)
        self.graph_fig.savefig(self.save_path)
        plt.close();
        
if __name__=='__main__':
    import os
    import numpy as np 
    import pandas as pd 
    import torch 
    import datasets 
    from transformers import AutoTokenizer,DataCollatorWithPadding
    from transformers import AutoModelForSequenceClassification 
    from transformers import TrainingArguments,Trainer 
    from transformers import EarlyStoppingCallback

    from tqdm import tqdm 
    from transformers import AdamW, get_scheduler
    from torchkeras.tools.transformers import VLogCallback 
    
    #================================================================================
    # 1,prepare data
    
    data_url = 'https://github.com/lyhue1991/torchkeras/releases/download/v3.7.2/waimai_10k.csv'
    data_file = 'waimai_10k.csv'

    if not os.path.exists(data_file):
        torch.hub.download_url_to_file(data_url,data_file)


    df = pd.read_csv("waimai_10k.csv")
    ds = datasets.Dataset.from_pandas(df)
    ds = ds.shuffle(42) 
    ds = ds.rename_columns({"review":"text","label":"labels"})

    tokenizer = AutoTokenizer.from_pretrained('bert-base-chinese') 

    ds_encoded = ds.map(lambda example:tokenizer(example["text"]),
                        remove_columns = ["text"],
                        batched=True)

    #train,val,test split
    ds_train_val,ds_test = ds_encoded.train_test_split(test_size=0.2).values()
    ds_train,ds_val = ds_train_val.train_test_split(test_size=0.2).values() 

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    dl_train = torch.utils.data.DataLoader(ds_train, batch_size=16, collate_fn = data_collator)
    dl_val = torch.utils.data.DataLoader(ds_val, batch_size=16,  collate_fn = data_collator)
    dl_test = torch.utils.data.DataLoader(ds_test, batch_size=16,  collate_fn = data_collator)

    for batch in dl_train:
        break
    print({k: v.shape for k, v in batch.items()})
    
    #================================================================================
    #2，define model
    
    model = AutoModelForSequenceClassification.from_pretrained(
        'bert-base-chinese',num_labels=2)


    #================================================================================
    #3，train model
    
    def compute_metrics(eval_preds):
        logits, labels = eval_preds
        preds = np.argmax(logits, axis=-1)
        accuracy = np.sum(preds==labels)/len(labels)
        precision = np.sum((preds==1)&(labels==1))/np.sum(preds==1)
        recall = np.sum((preds==1)&(labels==1))/np.sum(labels==1)
        f1  = 2*recall*precision/(recall+precision)
        return {"accuracy":accuracy,"precision":precision,"recall":recall,'f1':f1}

    training_args = TrainingArguments(
        output_dir = "bert_waimai",
        num_train_epochs = 2,
        logging_steps = 20,
        gradient_accumulation_steps = 10,
        evaluation_strategy="steps", #epoch

        metric_for_best_model='eval_f1',
        greater_is_better=True,

        report_to='none',
        load_best_model_at_end=True
    )

    callbacks = [EarlyStoppingCallback(early_stopping_patience=10),
                 VLogCallback()]

    trainer = Trainer(
        model,
        training_args,
        train_dataset=ds_train,
        eval_dataset=ds_val,
        compute_metrics=compute_metrics,
        callbacks = callbacks,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    trainer.train() 


    #================================================================================
    #4，evaluate model
    
    trainer.evaluate(ds_val)


    #================================================================================
    #5，use model
    
    from transformers import pipeline
    model.config.id2label = {0:"差评",1:"好评"}
    classifier = pipeline(task="text-classification",tokenizer = tokenizer,model=model.cpu())
    classifier("挺好吃的哦")

    #================================================================================
    #6，save model
    
    model.save_pretrained("waimai_10k_bert")
    tokenizer.save_pretrained("waimai_10k_bert")

    classifier = pipeline("text-classification",model="waimai_10k_bert")
    classifier(["味道还不错，下次再来","我去，吃了我吐了三天"])
