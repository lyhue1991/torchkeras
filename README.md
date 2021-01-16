# 1，Introduction

<!-- #region -->
The torchkeras library is a simple tool for training neural network in pytorch jusk like in a keras style. 😋😋

With torchkeras, You need not to write your training loop with many lines of code, all you need to do is just 

like this three steps as below:

(i) create your network and wrap it with torchkeras.Model like this: `model = torchkeras.Model(net)` 

(ii) compile your model to 	bind the loss function, the optimizer and the metrics function.

(iii) fit your model with the training data and validate data.

**This project seems somehow powerful, but the source code is very simple.**

**Actually, less than 200 lines of Python code.**

**If you want to understand or modify some details of this project, feel free to read and change the source code!!!**


<br/>

🍉🍉 **new feature**🍉🍉: torchkeras.LightModel 😋😋

it's more powerful than torchkeras.Model and with more flexibility and easier to use!

the tutorial of torchkeras.LightModel is here:

**[Use Pytorch-Lightning Like Keras ](./Tutorial.md)**


<br/>


<!-- #endregion -->

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
def accuracy(y_pred,y_true):
    y_pred = torch.where(y_pred>0.5,torch.ones_like(y_pred,dtype = torch.float32),
                      torch.zeros_like(y_pred,dtype = torch.float32))
    acc = torch.mean(1-torch.abs(y_true-y_pred))
    return acc

# if gpu is available, use gpu
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model.compile(loss_func = nn.BCELoss(),optimizer= torch.optim.Adam(model.parameters(),lr = 0.01),
             metrics_dict={"accuracy":accuracy},device = device)

dfhistory = model.fit(30,dl_train = dl_train,dl_val = dl_valid,log_step_freq = 20)
```

```
Start Training ...

================================================================================2020-06-21 20:40:23
{'step': 10, 'loss': 0.217, 'accuracy': 0.905}
{'step': 20, 'loss': 0.215, 'accuracy': 0.914}

 +-------+-------+----------+----------+--------------+
| epoch |  loss | accuracy | val_loss | val_accuracy |
+-------+-------+----------+----------+--------------+
|   1   | 0.212 |  0.914   |  0.186   |    0.927     |
+-------+-------+----------+----------+--------------+

================================================================================2020-06-21 20:40:23
{'step': 10, 'loss': 0.211, 'accuracy': 0.912}
{'step': 20, 'loss': 0.193, 'accuracy': 0.919}

 +-------+-------+----------+----------+--------------+
| epoch |  loss | accuracy | val_loss | val_accuracy |
+-------+-------+----------+----------+--------------+
|   2   | 0.194 |  0.919   |  0.188   |    0.935     |
+-------+-------+----------+----------+--------------+

================================================================================2020-06-21 20:40:23
{'step': 10, 'loss': 0.217, 'accuracy': 0.913}
{'step': 20, 'loss': 0.205, 'accuracy': 0.92}

 +-------+-------+----------+----------+--------------+
| epoch |  loss | accuracy | val_loss | val_accuracy |
+-------+-------+----------+----------+--------------+
|   3   | 0.195 |  0.921   |  0.176   |    0.931     |
+-------+-------+----------+----------+--------------+

================================================================================2020-06-21 20:40:23
{'step': 10, 'loss': 0.164, 'accuracy': 0.932}
{'step': 20, 'loss': 0.197, 'accuracy': 0.917}

 +-------+-------+----------+----------+--------------+
| epoch |  loss | accuracy | val_loss | val_accuracy |
+-------+-------+----------+----------+--------------+
|   4   | 0.197 |  0.917   |  0.178   |    0.935     |
+-------+-------+----------+----------+--------------+

================================================================================2020-06-21 20:40:24
{'step': 10, 'loss': 0.192, 'accuracy': 0.926}
{'step': 20, 'loss': 0.182, 'accuracy': 0.931}

 +-------+-------+----------+----------+--------------+
| epoch |  loss | accuracy | val_loss | val_accuracy |
+-------+-------+----------+----------+--------------+
|   5   | 0.193 |  0.924   |  0.188   |    0.928     |
+-------+-------+----------+----------+--------------+

================================================================================2020-06-21 20:40:44
{'step': 10, 'loss': 0.175, 'accuracy': 0.932}
{'step': 20, 'loss': 0.188, 'accuracy': 0.924}

 +-------+-------+----------+----------+--------------+
| epoch |  loss | accuracy | val_loss | val_accuracy |
+-------+-------+----------+----------+--------------+
|   97  | 0.184 |  0.923   |  0.176   |    0.935     |
+-------+-------+----------+----------+--------------+

================================================================================2020-06-21 20:40:44
{'step': 10, 'loss': 0.21, 'accuracy': 0.913}
{'step': 20, 'loss': 0.192, 'accuracy': 0.918}

 +-------+------+----------+----------+--------------+
| epoch | loss | accuracy | val_loss | val_accuracy |
+-------+------+----------+----------+--------------+
|   98  | 0.19 |  0.922   |  0.179   |    0.934     |
+-------+------+----------+----------+--------------+

================================================================================2020-06-21 20:40:45
{'step': 10, 'loss': 0.186, 'accuracy': 0.923}
{'step': 20, 'loss': 0.181, 'accuracy': 0.928}

 +-------+-------+----------+----------+--------------+
| epoch |  loss | accuracy | val_loss | val_accuracy |
+-------+-------+----------+----------+--------------+
|   99  | 0.182 |  0.926   |  0.178   |    0.938     |
+-------+-------+----------+----------+--------------+

================================================================================2020-06-21 20:40:45
{'step': 10, 'loss': 0.16, 'accuracy': 0.93}
{'step': 20, 'loss': 0.173, 'accuracy': 0.93}

 +-------+-------+----------+----------+--------------+
| epoch |  loss | accuracy | val_loss | val_accuracy |
+-------+-------+----------+----------+--------------+
|  100  | 0.185 |  0.925   |  0.174   |    0.936     |
+-------+-------+----------+----------+--------------+

================================================================================2020-06-21 20:40:45
Finished Training...
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
plot_metric(dfhistory,"loss")
```

![](./data/loss_curve.png)

```python
plot_metric(dfhistory,"accuracy")
```

![](./data/accuracy_curve.png)

```python

```

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

```python

```

### (6) save the model

```python
# save the model parameters

torch.save(model.state_dict(), "model_parameter.pkl")

model_clone = torchkeras.Model(Net())
model_clone.load_state_dict(torch.load("model_parameter.pkl"))

model_clone.compile(loss_func = nn.BCELoss(),optimizer= torch.optim.Adam(model.parameters(),lr = 0.01),
             metrics_dict={"accuracy":accuracy})

model_clone.evaluate(dl_valid)
```

```
{'val_loss': 0.17422042911251387, 'val_accuracy': 0.9358333299557368}
```
