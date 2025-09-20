from torchkeras import VLog
class VLogCallback:
    def __init__(self,epochs,monitor_metric,monitor_mode):
        self.vlog = VLog(epochs,monitor_metric,monitor_mode)
        
    def on_train_batch_end(self,trainer):
        self.vlog.log_step(trainer.label_loss_items(trainer.tloss, prefix='train'))

    def on_fit_epoch_end(self,trainer):
        metrics = {k.split('/')[-1]:v for k,v in trainer.metrics.items() if 'loss' not in k}
        self.vlog.log_epoch(metrics)

        
if __name__=='__main__':
    import os,shutil
    import torch 
    from ultralytics import YOLO 
    from torchkeras.tools.ultralytics import VLogCallback

    # 1,prepare data
    data_url = 'https://github.com/lyhue1991/torchkeras/releases/download/v3.7.2/cats_vs_dogs.zip'
    data_file = 'cats_vs_dogs.zip'

    if not os.path.exists(data_file):
        torch.hub.download_url_to_file(data_url,data_file)
        shutil.unpack_archive(data_file,'datasets')

    #================================================================================
    # 2,define model

    model = YOLO(model = 'yolov8n-cls.pt')


    #================================================================================
    
    # 3,train model
    epochs = 100
    vlog_cb = VLogCallback(epochs = epochs,
                           monitor_metric='accuracy_top1',
                           monitor_mode='max')
    callbacks = {
        "on_train_batch_end": vlog_cb.on_train_batch_end,
        "on_fit_epoch_end": vlog_cb.on_fit_epoch_end
    }
    for event,func in callbacks.items():
        model.add_callback(event,func)
    vlog_cb.vlog.log_start()
    results = model.train(data='datasets/cats_vs_dogs', 
                          epochs=epochs, workers=4)     
    vlog_cb.vlog.log_end()
