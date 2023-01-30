
# Pytorch❤️Keras

The torchkeras library is a simple tool for training neural network in pytorch jusk in a keras style. 😋😋


 <br>

 <div>
    </a>
     <a href="https://www.kaggle.com/lyhue1991/torchkeras-ddp-tpu-examples"><img src="https://kaggle.com/static/images/open-in-kaggle.svg" alt="Open In Kaggle"></a>
  </div>
 <br>


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
                              optimizer= torch.optim.Adam(net.parameters(),lr = 0.001),
                              metrics_dict = {"acc":torchmetrics.Accuracy(task='binary')}
                             )
dfhistory=model.fit(train_data=dl_train, 
                    val_data=dl_val, 
                    epochs=20, 
                    patience=3, 
                    ckpt_path='checkpoint.pt',
                    monitor="val_acc",
                    mode="max")

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
|multi-gpus training(ddp) |   ✅   |✅    |
|tensorboard callback |   ✅   |✅    |
|pretty wandb callback |  ✅ 👍🏻   |  ❌ |
|other callbacks from pytorch_lightning |   ❌  |✅    |
|simple code |  ✅   |❌    |

```python

```

### 3, Basic Examples 


You can follow these full examples to get started with torchkeras.

Have fun!😋😋

* ① [**torchkeras.KerasModel example**](./1，kerasmodel_example.ipynb) 🔥🔥

* ② [**torchkeras.KerasModel with wandb demo**](./2，kerasmodel_wandb_demo.ipynb) 🔥🔥🔥

* ③ [**torchkeras.KerasModel with tensorboard example**](./3，kerasmodel_tensorboard_demo.ipynb)

* ④ [**torchkeras.KerasModel  ddp tpu examples**](https://www.kaggle.com/code/lyhue1991/torchkeras-ddp-tpu-examples)

* ⑤ [**torchkeras.LightModel example**](./5，lightmodel_example.ipynb)

* ⑥ [**torchkeras.LightModel  with tensorboard example**](./6，lightmodel_tensorboard_demo.ipynb)





**If you want to understand or modify some details of this project, feel free to read and change the source code!!!**

Any other questions, you can contact the author form the wechat official account below:

**算法美食屋** 


![](https://tva1.sinaimg.cn/large/e6c9d24egy1h41m2zugguj20k00b9q46.jpg)

