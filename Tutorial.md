### Use Pytorch-Lightning Like Keras 


Lightning disentangles PyTorch code to decouple the science from the engineering.

Lightning is designed with these principles in mind:

Principle 1: Enable maximal flexibility.

Principle 2: Abstract away unnecessary boilerplate, but make it accessible when needed. 

Principle 3: Systems should be self-contained (ie: optimizers, computation code, etc). 

Principle 4: Deep learning code should be organized into 4 distinct categories.

* Research code (the LightningModule).
* Engineering code (you delete, and is handled by the Trainer).
* Non-essential research code (logging, etc... this goes in Callbacks).
* Data (use PyTorch Dataloaders or organize them into a LightningDataModule).

Once you do this, you can train on multiple-GPUs, TPUs, CPUs and even in 16-bit precision without changing your code!

What's more, we can use Pytorch-Lightning to implement the keras-style network trainning and evaluating.

Below is an example, the class `torchkeras.nightModel` is  very similar to the class `torchkeras.Model`.

While torchkeras.nightModel borrows a lot of power from the Pytorch-Lightning Module, like:

* train with multi-gpus 

* train with tpu or tpus

* use many callbacks such as ModelCheckpoint, EarlyStopping ......

* auto model and parameters save 

* use lr_schedule freely

And what's more, it enables much more flexibility and easier to use.

All you need it to write a `shared_step` function and return the loss and metrics dict.



```python

```

## 2, Use Example 

```python
import numpy as np 
import pandas as pd 
from matplotlib import pyplot as plt
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset,DataLoader,TensorDataset
import pytorch_lightning as pl 
from torchkeras import LightModel 
import datetime

```

### (1)，prepare data 

```python
%matplotlib inline
%config InlineBackend.figure_format = 'svg'

#number of samples
n_positive,n_negative = 2000,2000

#positive samples
r_p = 5.0 + torch.normal(0.0,1.0,size = [n_positive,1]) 
theta_p = 2*np.pi*torch.rand([n_positive,1])
Xp = torch.cat([r_p*torch.cos(theta_p),r_p*torch.sin(theta_p)],axis = 1)
Yp = torch.ones_like(r_p)

#negative samples
r_n = 8.0 + torch.normal(0.0,1.0,size = [n_negative,1]) 
theta_n = 2*np.pi*torch.rand([n_negative,1])
Xn = torch.cat([r_n*torch.cos(theta_n),r_n*torch.sin(theta_n)],axis = 1)
Yn = torch.zeros_like(r_n)

#concat positive and negative samples
X = torch.cat([Xp,Xn],axis = 0)
Y = torch.cat([Yp,Yn],axis = 0)


#visual samples
plt.figure(figsize = (6,6))
plt.scatter(Xp[:,0],Xp[:,1],c = "r")
plt.scatter(Xn[:,0],Xn[:,1],c = "g")
plt.legend(["positive","negative"]);


```

![](./data/input_data.png)

```python
# split samples into train and valid data.
ds = TensorDataset(X,Y)
ds_train,ds_valid = torch.utils.data.random_split(ds,[int(len(ds)*0.7),len(ds)-int(len(ds)*0.7)])
dl_train = DataLoader(ds_train,batch_size = 100,shuffle=True,num_workers=4)
dl_valid = DataLoader(ds_valid,batch_size = 100,num_workers=4)

```

```python

```

### (2)，create the  model

```python
class Net(nn.Module):  
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2,4)
        self.fc2 = nn.Linear(4,8) 
        self.fc3 = nn.Linear(8,1)
        
    def forward(self,x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        y = nn.Sigmoid()(self.fc3(x))
        return y       
```

```python
class Model(LightModel):
    
    def shared_step(self,batch)->dict:
        x, y = batch
        prediction = self(x)
        loss = nn.BCELoss()(prediction,y)
        preds = torch.where(prediction>0.5,torch.ones_like(prediction),torch.zeros_like(prediction))
        acc = pl.metrics.functional.accuracy(preds, y)
        # attention: there must be a key of "loss" in the returned dict 
        dic = {"loss":loss,"acc":acc} 
        return dic
    
    #optimizer,and optional lr_scheduler
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-2)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.0001)
        return {"optimizer":optimizer,"lr_scheduler":lr_scheduler}
 
```

<!-- #region -->
The class Model can alse defined like below without subclassing from torchkeras.LightModel.

```python 

class Model(pl.LightningModule):
    
    #loss,and optional metrics
    def shared_step(self,batch)->dict:
        x, y = batch
        prediction = self(x)
        loss = nn.BCELoss()(prediction,y)
        preds = torch.where(prediction>0.5,torch.ones_like(prediction),torch.zeros_like(prediction))
        acc = pl.metrics.functional.accuracy(preds, y)
        # attention: there must be a key of "loss" in the returned dict 
        dic = {"loss":loss,"acc":acc} 
        return dic
    
    #optimizer,and optional lr_scheduler
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-2)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.0001)
        return {"optimizer":optimizer,"lr_scheduler":lr_scheduler}
    
    #================================================================================
    # Codes Below should not be changed!!!
    #================================================================================
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
        
```
<!-- #endregion -->

```python
pl.seed_everything(1234)
net = Net()
model = Model(net)

from torchkeras import summary 
summary(model,input_shape =(2,))

```

### 3， train the model

```python

```

```python


ckpt_cb = pl.callbacks.ModelCheckpoint(monitor='val_loss')

# set gpus=0 will use cpu，
# set gpus=1 will use 1 gpu
# set gpus=2 will use 2gpus 
# set gpus = -1 will use all gpus 
# you can also set gpus = [0,1] to use the  given gpus

trainer = pl.Trainer(max_epochs=10,gpus = 0, callbacks=[ckpt_cb]) 

trainer.fit(model,dl_train,dl_valid)


```

```
================================================================================2021-01-16 13:37:20
epoch =  0
{'val_loss': 0.6791770458221436, 'val_acc': 0.5483333468437195}
{'acc': 0.5042856931686401, 'loss': 0.6874533891677856}

================================================================================2021-01-16 13:37:21
epoch =  1
{'val_loss': 0.6675646305084229, 'val_acc': 0.5849999785423279}
{'acc': 0.5364286303520203, 'loss': 0.6739903092384338}

================================================================================2021-01-16 13:37:22
epoch =  2
{'val_loss': 0.645446240901947, 'val_acc': 0.6625000238418579}
{'acc': 0.5742858052253723, 'loss': 0.6572595834732056}

================================================================================2021-01-16 13:37:23
epoch =  3
{'val_loss': 0.5887902975082397, 'val_acc': 0.6899999976158142}
{'acc': 0.6271429061889648, 'loss': 0.6197047829627991}

================================================================================2021-01-16 13:37:24
epoch =  4
{'val_loss': 0.4813719689846039, 'val_acc': 0.7766666412353516}
{'acc': 0.733571469783783, 'loss': 0.5395488142967224}

================================================================================2021-01-16 13:37:26
epoch =  5
{'val_loss': 0.3596847355365753, 'val_acc': 0.8833332657814026}
{'acc': 0.8160714507102966, 'loss': 0.4321393072605133}

================================================================================2021-01-16 13:37:27
epoch =  6
{'val_loss': 0.28268730640411377, 'val_acc': 0.9016666412353516}
{'acc': 0.8853570818901062, 'loss': 0.3260730803012848}

================================================================================2021-01-16 13:37:28
epoch =  7
{'val_loss': 0.22369669377803802, 'val_acc': 0.9141666293144226}
{'acc': 0.8949999809265137, 'loss': 0.2615409791469574}

================================================================================2021-01-16 13:37:29
epoch =  8
{'val_loss': 0.21904177963733673, 'val_acc': 0.9091666340827942}
{'acc': 0.9039285778999329, 'loss': 0.23385460674762726}

================================================================================2021-01-16 13:37:30
epoch =  9
{'val_loss': 0.2023438811302185, 'val_acc': 0.9200000166893005}
{'acc': 0.909285843372345, 'loss': 0.22031544148921967}
```

```python
# visual the results
fig, (ax1,ax2) = plt.subplots(nrows=1,ncols=2,figsize = (12,5))
ax1.scatter(Xp[:,0],Xp[:,1], c="r")
ax1.scatter(Xn[:,0],Xn[:,1],c = "g")
ax1.legend(["positive","negative"]);
ax1.set_title("y_true")

Xp_pred = X[torch.squeeze(model.forward(X)>=0.5)]
Xn_pred = X[torch.squeeze(model.forward(X)<0.5)]

ax2.scatter(Xp_pred[:,0],Xp_pred[:,1],c = "r")
ax2.scatter(Xn_pred[:,0],Xn_pred[:,1],c = "g")
ax2.legend(["positive","negative"]);
ax2.set_title("y_pred")

```

![](./data/training_result.png)

```python

```

### 4，evaluate model 

```python
import pandas as pd 

history = model.history
dfhistory = pd.DataFrame(history) 
dfhistory 
```

```python
%matplotlib inline
%config InlineBackend.figure_format = 'svg'

import matplotlib.pyplot as plt

def plot_metric(dfhistory, metric):
    train_metrics = dfhistory[metric]
    val_metrics = dfhistory['val_'+metric]
    epochs = range(1, len(train_metrics) + 1)
    plt.plot(epochs, train_metrics, 'bo--')
    plt.plot(epochs, val_metrics, 'ro-')
    plt.title('Training and validation '+ metric)
    plt.xlabel("Epochs")
    plt.ylabel(metric)
    plt.legend(["train_"+metric, 'val_'+metric])
    plt.show()
    
```

```python
import prettytable
print(prettytable.__version__)
```

```python
plot_metric(dfhistory,"loss")
```

```python
plot_metric(dfhistory,"acc")
```

```python

```

```python

results = trainer.test(model, test_dataloaders=dl_valid, verbose = False)
print(results[0])

```

```
{'test_loss': 0.24610649049282074, 'test_acc': 0.9100000262260437}
```

```python

```

### 5，use the model 

```python
def predict(model,dl):
    model.eval()
    result = torch.cat([model.forward(t[0].to(model.device)) for t in dl])
    return(result.data)

result = predict(model,dl_valid)
```

```python
result 
```

```
tensor([[0.9564],
        [0.0170],
        [0.9953],
        ...,
        [0.0384],
        [0.9686],
        [0.9475]])
```

```python

```

### 6，save the model 


The model's structure and parameters value is saved  int the lightning_logs path. 


```python
print(ckpt_cb.best_model_score)
model.load_from_checkpoint(ckpt_cb.best_model_path)

best_net  = model.net
torch.save(best_net.state_dict(),"net.pt")

```

```python
net_clone = Net()
net_clone.load_state_dict(torch.load("net.pt"))
model_clone = Model(net_clone)
trainer = pl.Trainer()
result = trainer.test(model_clone,test_dataloaders=dl_valid, verbose = False) 

print(result)

```

```
[{'test_loss': 0.24610649049282074, 'test_acc': 0.9100000262260437}]
```

```python

```

```python

```
