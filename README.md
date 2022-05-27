# 1，Introduction


The torchkeras library is a simple tool for training neural network in pytorch jusk like in a keras style. 😋😋

With torchkeras, You need not to write your training loop with many lines of code, all you need to do is just 

like this three steps as below:

(i) create your network and wrap it and the loss_fn together with torchkeras.KerasModel like this: `model = torchkeras.KerasModel(net,loss_fn)` 

(ii) fit your model with the training data and validate data.

**This project seems somehow powerful, but the source code is very simple.**

**Actually, less than 200 lines of Python code.**

**If you want to understand or modify some details of this project, feel free to read and change the source code!!!**





# 2,  Use example


You can install torchkeras using pip:
`pip install torchkeras`


Here is a complete examples using torchkeras! 

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

### (1) prepare data 

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
dl_train = DataLoader(ds_train,batch_size = 100,shuffle=True,num_workers=2)
dl_valid = DataLoader(ds_valid,batch_size = 100,num_workers=2)
```

### (2) create the  model

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
        
net = Net()

### Attention here
model = torchkeras.Model(net)
model.summary(input_shape =(2,))
```

```
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Linear-1                    [-1, 4]              12
            Linear-2                    [-1, 8]              40
            Linear-3                    [-1, 1]               9
================================================================
Total params: 61
Trainable params: 61
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.000008
Forward/backward pass size (MB): 0.000099
Params size (MB): 0.000233
Estimated Total Size (MB): 0.000340
----------------------------------------------------------------
```


### (3) train the model

```python
# define metric
def accuracy(y_pred, y_true):
    y_pred = torch.where(y_pred>0.5,torch.ones_like(y_pred,dtype = torch.float32),
                      torch.zeros_like(y_pred,dtype = torch.float32))
    acc = torch.mean(1-torch.abs(y_true-y_pred))
    return acc


def mse(y_pred, y_true):
    return torch.sqrt(torch.mean((y_true - y_pred) ** 2))


# if gpu is available, use gpu
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model.compile(loss_func = nn.BCELoss(),optimizer= torch.optim.Adam(model.parameters(),lr = 0.01),
             metrics_dict={accuracy, mse},device = device)

dfhistory=model.fit(epochs=10, train_data=dl_train, val_data=dl_valid, patience=5, monitor="val_loss", save_path="save_model.pkl", verbose=True)
```

```
Epoch 1 / 10
[========================================] 100%	loss: 0.6570    accuracy: 0.5525    mse: 0.4824    val_loss: 0.6188    val_accuracy: 0.6167    val_mse: 0.4638

Validation loss decreased (inf --> 0.618847).  Saving model ...
Epoch 2 / 10
[========================================] 100%	loss: 0.5877    accuracy: 0.6857    mse: 0.4476    val_loss: 0.5518    val_accuracy: 0.6950    val_mse: 0.4315

Validation loss decreased (0.618847 --> 0.551810).  Saving model ...
Epoch 3 / 10
[========================================] 100%	loss: 0.4949    accuracy: 0.8079    mse: 0.3984    val_loss: 0.4342    val_accuracy: 0.8558    val_mse: 0.3645

Validation loss decreased (0.551810 --> 0.434237).  Saving model ...
Epoch 4 / 10
[========================================] 100%	loss: 0.3819    accuracy: 0.8682    mse: 0.3359    val_loss: 0.3284    val_accuracy: 0.9117    val_mse: 0.3023

Validation loss decreased (0.434237 --> 0.328433).  Saving model ...
Epoch 5 / 10
[========================================] 100%	loss: 0.2942    accuracy: 0.9007    mse: 0.2882    val_loss: 0.2541    val_accuracy: 0.9092    val_mse: 0.2649

Validation loss decreased (0.328433 --> 0.254060).  Saving model ...
Epoch 6 / 10
[========================================] 100%	loss: 0.2441    accuracy: 0.9104    mse: 0.2627    val_loss: 0.2311    val_accuracy: 0.9125    val_mse: 0.2561

Validation loss decreased (0.254060 --> 0.231079).  Saving model ...
Epoch 7 / 10
[========================================] 100%	loss: 0.2247    accuracy: 0.9100    mse: 0.2542    val_loss: 0.2218    val_accuracy: 0.9083    val_mse: 0.2546

Validation loss decreased (0.231079 --> 0.221847).  Saving model ...
Epoch 8 / 10
[========================================] 100%	loss: 0.2091    accuracy: 0.9164    mse: 0.2441    val_loss: 0.2084    val_accuracy: 0.9192    val_mse: 0.2441

Validation loss decreased (0.221847 --> 0.208386).  Saving model ...
Epoch 9 / 10
[========================================] 100%	loss: 0.1972    accuracy: 0.9218    mse: 0.2366    val_loss: 0.2032    val_accuracy: 0.9175    val_mse: 0.2435

Validation loss decreased (0.208386 --> 0.203234).  Saving model ...
Epoch 10 / 10
[========================================] 100%	loss: 0.1940    accuracy: 0.9204    mse: 0.2367    val_loss: 0.2058    val_accuracy: 0.9167    val_mse: 0.2445

EarlyStopping counter: 1 out of 5
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


### (4) evaluate the model

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
plot_metric(dfhistory,"loss")
```

![](./data/loss_curve.png)

```python
plot_metric(dfhistory,"accuracy")
```

![](./data/accuracy_curve.png)


```python
model.evaluate(dl_valid)
```

```
{'val_loss': 0.13576620258390903, 'val_accuracy': 0.9441666702429453}
```


### (5) use the model

```python
model.predict(dl_valid)[0:10]
```

```
tensor([[0.8767],
        [0.0154],
        [0.9976],
        [0.9990],
        [0.9984],
        [0.0071],
        [0.3529],
        [0.4061],
        [0.9938],
        [0.9997]])
```

```python
for features,labels in dl_valid:
    with torch.no_grad():
        predictions = model.forward(features)
        print(predictions[0:10])
    break
```

```
tensor([[0.9979],
        [0.0011],
        [0.9782],
        [0.9675],
        [0.9653],
        [0.9906],
        [0.1774],
        [0.9994],
        [0.9178],
        [0.9579]])
```

### (6) save the model

```python
# save the model parameters

model_clone = torchkeras.Model(Net())
model_clone.load_state_dict(torch.load("save_model.pkl"))

model_clone.compile(loss_func = nn.BCELoss(),optimizer= torch.optim.Adam(model.parameters(),lr = 0.01),
             metrics_dict={"accuracy":accuracy})

model_clone.evaluate(dl_valid)
```

```
{'val_loss': 0.17422042911251387, 'val_accuracy': 0.9358333299557368}
```
