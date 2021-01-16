import torch 
import pytorch_lightning as pl 
import datetime

class LightModel(pl.LightningModule):
    
    def __init__(self,net=None):
        super().__init__()
        self.net = net
        self.save_hyperparameters("net")
        self.history = {}
        
    def forward(self,x):
        if self.net:
            return self.net.forward(x)
        else:
            raise NotImplementedError
    
    #loss,and optional metrics
    def shared_step(self,batch)->dict:
        raise NotImplementedError
    
    #optimizer,and optional lr_scheduler
    def configure_optimizers(self):
        raise NotImplementedError
        
    #================================================================================
    # Codes Below should not be changed!!!
    #================================================================================
    
    def training_step(self, batch, batch_idx):
        dic = self.shared_step(batch)
        self.log_dict(dic)
        return dic

    def validation_step(self, batch, batch_idx):
        dic = self.shared_step(batch)
        val_dic = {"val_"+k:v for k,v in dic.items()}
        self.log_dict(val_dic)
        return val_dic
    
    def test_step(self, batch, batch_idx):
        dic = self.shared_step(batch)
        test_dic = {"test_"+k:v for k,v in dic.items()}
        self.log_dict(test_dic)
        return test_dic
    
    def make_epoch_metrics(self,step_outputs):
        metrics = {}
        for dic in step_outputs:
            for k,v in dic.items():
                metrics[k] = metrics.get(k,[])+[v]
            
        for k,v in metrics.items():
            metrics[k] = (sum(metrics[k])/(len(metrics[k]))).item()
            
        for k,v in metrics.items():
            self.history[k] = self.history.get(k,[])+[v]
        return metrics
        
    def training_epoch_end(self, training_step_outputs):
        train_metrics = self.make_epoch_metrics(training_step_outputs)
        self.history["epoch"] = self.history.get("epoch",[])+[self.trainer.current_epoch]
        self.print(train_metrics)

    def validation_epoch_end(self, validation_step_outputs):
        val_metrics = self.make_epoch_metrics(validation_step_outputs)
        self.print_bar()
        self.print("epoch = ", self.trainer.current_epoch)
        self.print(val_metrics)
        
    def on_train_end(self):
        for k,v in self.history.items():
            if "val_" in k and len(v)>len(self.history.get("epoch",[])):
                self.history[k] = self.history[k][1:]
        
    def print_bar(self): 
        nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        self.print("\n"+"="*80 + "%s"%nowtime)
