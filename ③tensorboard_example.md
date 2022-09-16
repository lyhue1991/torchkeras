# TensorBoard Example


You can install torchkeras using pip:
`pip install torchkeras`

Here is a complete examples using torchkeras.LightModel  with visualization of tensorboad.



```python
import sys 
sys.path.append("..")
```

```python
import numpy as np 
import pandas as pd 
from matplotlib import pyplot as plt
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset,DataLoader,TensorDataset

import torchkeras #Attention this line 

```

### 1, prepare data 

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
ds_train,ds_val = torch.utils.data.random_split(ds,[int(len(ds)*0.7),len(ds)-int(len(ds)*0.7)])
dl_train = DataLoader(ds_train,batch_size = 200,shuffle=True,num_workers=2)
dl_val = DataLoader(ds_val,batch_size = 200,num_workers=2)

```

```python
for features,labels in dl_train:
    break
print(features.shape)
print(labels.shape)

```

```python

```

### 2, create the  model

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
        y = self.fc3(x) #don't need nn.Sigmoid()
        return y
    
net = Net() 


```

```python

```

```python
from torchkeras.metrics import Accuracy 

loss_fn = nn.BCEWithLogitsLoss()
metric_dict = {"acc":Accuracy()}

optimizer = torch.optim.Adam(net.parameters(), lr=0.03)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.0001)

model = torchkeras.LightModel(net,
                   loss_fn = loss_fn,
                   metrics_dict= metric_dict,
                   optimizer = optimizer,
                   lr_scheduler = lr_scheduler,
                  )       

from torchkeras import summary
summary(model,input_data=features);

```

```
--------------------------------------------------------------------------
Layer (type)                            Output Shape              Param #
==========================================================================
Linear-1                                     [-1, 4]                   12
Linear-2                                     [-1, 8]                   40
Linear-3                                     [-1, 1]                    9
==========================================================================
Total params: 61
Trainable params: 61
Non-trainable params: 0
--------------------------------------------------------------------------
Input size (MB): 0.000069
Forward/backward pass size (MB): 0.000099
Params size (MB): 0.000233
Estimated Total Size (MB): 0.000401
--------------------------------------------------------------------------


```


### 3, train the model

```python
import pytorch_lightning as pl  
from torchkeras.callbacks import TensorBoard
```

```python
#1，设置回调函数
model_ckpt = pl.callbacks.ModelCheckpoint(
    monitor='val_loss',
    save_top_k=1,
    mode='min'
)

early_stopping = pl.callbacks.EarlyStopping(monitor = 'val_loss',
                           patience=5,
                           mode = 'min'
                          )

tensorboard = TensorBoard(
    save_dir='tb_logs',
    model_name='cnn',
    log_weight=True,
    log_weight_freq=2,
    log_graph=True,
    example_input_array=features,
    log_hparams=True,
    hparams_dict=None,
)

#2，设置训练参数

# gpus=0 则使用cpu训练，gpus=1则使用1个gpu训练，gpus=2则使用2个gpu训练，gpus=-1则使用所有gpu训练，
# gpus=[0,1]则指定使用0号和1号gpu训练， gpus="0,1,2,3"则使用0,1,2,3号gpu训练
# tpus=1 则使用1个tpu训练
trainer = pl.Trainer(logger=True,
                     min_epochs=3,max_epochs=10,
                     gpus=0,
                     callbacks = [model_ckpt,early_stopping,tensorboard],
                     enable_progress_bar = True) 

#断点续训
#trainer = pl.Trainer(resume_from_checkpoint='./lightning_logs/version_31/checkpoints/epoch=02-val_loss=0.05.ckpt')

##4，启动训练循环
trainer.fit(model,dl_train,dl_val)
```

```python

```

### 4, Monitor from TensorBoard



The callback TensorBoard saves logs at the directory  'tb_logs'.

We can monitor and analysis training process using TensorBoard now.


!tensorboard --logdir="./tb_logs" --bind_all --port=6006

```python
from tensorboard import notebook
notebook.list() 
notebook.start("--logdir ./tb_logs")

```

#### metrics

![](https://tva1.sinaimg.cn/large/e6c9d24egy1h412vlgpqdj20n40cmaaf.jpg)


#### graphs

![](https://tva1.sinaimg.cn/large/e6c9d24egy1h4130c9g6lj20d90dd0st.jpg)

```python

```

#### histograms 

![](https://tva1.sinaimg.cn/large/e6c9d24egy1h4132au1scj20e709xjri.jpg) 
