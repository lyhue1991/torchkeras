import numpy as np 
import pandas as pd 
from matplotlib import pyplot as plt
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset,DataLoader
import torchkeras 

# ### 1, prepare data 

import torchvision 
from torchvision import transforms

transform = transforms.Compose([transforms.ToTensor()])
ds_train = torchvision.datasets.MNIST(root="mnist/",train=True,download=True,transform=transform)
ds_train = torch.utils.data.Subset(ds_train,range(0,len(ds_train),20))
ds_val = torchvision.datasets.MNIST(root="mnist/",train=False,download=True,transform=transform)
ds_val = torch.utils.data.Subset(ds_val,range(0,len(ds_val),20))
dl_train =  torch.utils.data.DataLoader(ds_train, batch_size=128, shuffle=True, num_workers=0)
dl_val =  torch.utils.data.DataLoader(ds_val, batch_size=128, shuffle=False, num_workers=0)


for features,labels in dl_train:
    break
print(features.shape)
print(labels.shape)



# ### 2, create the  model


def create_net():
    net = nn.Sequential()
    net.add_module("conv1",nn.Conv2d(in_channels=1,out_channels=32,kernel_size = 3))
    net.add_module("pool1",nn.MaxPool2d(kernel_size = 2,stride = 2))
    net.add_module("conv2",nn.Conv2d(in_channels=32,out_channels=64,kernel_size = 5))
    net.add_module("pool2",nn.MaxPool2d(kernel_size = 2,stride = 2))
    net.add_module("dropout",nn.Dropout2d(p = 0.1))
    net.add_module("adaptive_pool",nn.AdaptiveMaxPool2d((1,1)))
    net.add_module("flatten",nn.Flatten())
    net.add_module("linear1",nn.Linear(64,32))
    net.add_module("relu",nn.ReLU())
    net.add_module("linear2",nn.Linear(32,10))
    return net


class Accuracy(nn.Module):
    def __init__(self):
        super().__init__()

        self.correct = nn.Parameter(torch.tensor(0.0),requires_grad=False)
        self.total = nn.Parameter(torch.tensor(0.0),requires_grad=False)

    def forward(self, preds: torch.Tensor, targets: torch.Tensor):
        preds = preds.argmax(dim=-1)
        m = (preds == targets).sum()
        n = targets.shape[0] 
        self.correct += m 
        self.total += n
        
        return m/n

    def compute(self):
        return self.correct.float() / self.total 
    
    def reset(self):
        self.correct -= self.correct
        self.total -= self.total
        

net = create_net()


model = torchkeras.KerasModel(net,
      loss_fn = nn.CrossEntropyLoss(),
      optimizer= torch.optim.Adam(net.parameters(),lr=0.002),
      metrics_dict = {"acc":Accuracy()}
    )

from torchkeras import summary
summary(model,input_data=features);


# ### 3, train the model

ckpt_path='checkpoint.pt'
#model.load_ckpt(ckpt_path) #load trained ckpt and continue training
dfhistory=model.fit(train_data=dl_train, 
                    val_data=dl_val, 
                    epochs=100, 
                    patience=10, 
                    monitor="val_acc",
                    mode="max",
                    ckpt_path=ckpt_path,
                    plot=True,
                    wandb=False
                   )

model.evaluate(dl_val,quiet=False)


# ### 5, save the model

net_clone = create_net() 

model_clone = torchkeras.KerasModel(net_clone,loss_fn = nn.CrossEntropyLoss(),
             optimizer= torch.optim.Adam(net_clone.parameters(),lr = 0.001),
             metrics_dict={"acc":Accuracy()})

model_clone.load_ckpt("checkpoint.pt")