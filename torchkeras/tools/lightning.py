from torchkeras import VLog
from pytorch_lightning.callbacks import Callback
class VLogCallback(Callback):
    def __init__(self, monitor_metric='val_loss', monitor_mode='min'):
        super().__init__()
        self.monitor_metric = monitor_metric
        self.monitor_mode = monitor_mode 
    
    def on_fit_start(self, trainer, pl_module):
        self.vlog = VLog(trainer.max_epochs,
                         monitor_metric=self.monitor_metric, 
                         monitor_mode=self.monitor_mode
                        ) 
        self.vlog.log_start() 
        
    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.current_epoch>0:
            row_data = {k:v for k,v in trainer.logged_metrics.items() 
                        if k.split('_')[-1]==self.monitor_metric.split('_')[-1]
                       }
            self.vlog.log_epoch(row_data) 
            
    def on_fit_end(self,trainer,pl_module):
        self.vlog.log_end()