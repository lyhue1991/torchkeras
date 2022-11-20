
# Pytorch❤️Keras

The torchkeras library is a simple tool for training neural network in pytorch jusk in a keras style. 😋😋





## 1, Introduction

<!-- #region -->
With torchkeras, You need not to write your training loop with many lines of code, all you need to do is just 

like these two steps as below:


(i) create your network and wrap it and the loss_fn together with torchkeras.KerasModel like this: 
`model = torchkeras.KerasModel(net,loss_fn=nn.BCEWithLogitsLoss())` a metrics_dict parameter is optional.

(ii) fit your model with the training data and validate data.
<!-- #endregion -->

<!-- #region -->
The main code of use torchkeras is like below.

```python
import torch 
import torchkeras

#use torchkeras.KerasModel 
model = torchkeras.KerasModel(net,
                              loss_fn = nn.BCEWithLogitsLoss(),
                              optimizer= torch.optim.Adam(net.parameters(),lr = 0.03),
                              metrics_dict = {"acc":torchmetrics.metrics.Accuracy()}
                             )
dfhistory=model.fit(epochs=30, train_data=dl_train, 
                    val_data=dl_val, patience=3, 
                    monitor="val_acc",mode="max",
                    ckpt_path='checkpoint.pt')

```


**This project seems somehow powerful, but the source code is very simple.**

**Actually, less than 200 lines of Python code.**

**If you want to understand or modify some details of this project, feel free to read and change the source code!!!**
<!-- #endregion -->

```python

```

## 2, Features 

<!-- #region -->
Besides the basic torchkeras.KerasModel, another much more powerful class torchkeras.LightModel is created to support many other features.


The KerasModel is much simpler, and is recommended for beginner users.

The LightModel borrows many features from the library pytorch_lightning and shows a best practice.


Although different, the usage of torchkeras.KerasModel and  torchkeras.LightModel is very similar.


<!-- #endregion -->

|features| torchkeras.KerasModel     |  torchkeras.LightModel   | 
|----:|:-------------------------:|:-----------:|
|progress bar | ✅    |✅    |
|early stopping | ✅    |✅    |
|metrics from torchmetrics | ✅    |✅    |
|gpu training | ✅    |✅    |
|mac m1 training | ✅    |✅    |
|multi-gpus training |   ❌  |✅    |
|tensorboard callback |   ❌  |✅    |
|simple source code|   ✅   |❌  |

 

```python

```

### 3, Basic Examples 


You can follow these full examples to get started with torchkeras.

Have fun!😋😋

* ① [**torchkeras.KerasModel example**](./1，kerasmodel_example.ipynb)

* ② [**torchkeras.LightModel example**](./2，lightmodel_example.ipynb)

* ③ [**torchkeras.LightModel  with tensorboard example**](./3，tensorboard_example.ipynb)



**If you want to understand or modify some details of this project, feel free to read and change the source code!!!**

Any other questions, you can contact the author form the wechat official account below:

**算法美食屋** 


![](https://tva1.sinaimg.cn/large/e6c9d24egy1h41m2zugguj20k00b9q46.jpg)

