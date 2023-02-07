
# Pytorch❤️Keras

The torchkeras library is a simple tool for training neural network in pytorch jusk in a keras style. 😋😋


torchkeras ❤️ wandb: https://wandb.ai/lyhue1991/mnist_torchkeras

<br><div></a><a href="https://www.kaggle.com/lyhue1991/kerasmodel-wandb-example"><img src="https://kaggle.com/static/images/open-in-kaggle.svg" alt="Open In Kaggle"></a></div><br>




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

**Actually, only about 200 lines of Python code.**

**If you want to understand or modify some details of this project, feel free to read and change the source code!!!**
<!-- #endregion -->

```python

```

## 2, Features 


Besides the basic torchkeras.KerasModel, another powerful class torchkeras.LightModel is created to support pytorch_lightning training style.

The KerasModel is much simpler, and is recommended for beginner users.

The LightModel borrows many features from the library pytorch_lightning and shows a best practice.

Although different, the usage of torchkeras.KerasModel and  torchkeras.LightModel is very similar.




|features| torchkeras.KerasModel 🔥🔥🔥    |  torchkeras.LightModel   | 
|----:|:-------------------------:|:-----------:|
|progress bar | ✅    |✅    |
|early stopping | ✅    |✅    |
|metrics from torchmetrics | ✅    |✅    |
|gpu training | ✅    |✅    |
|multi-gpus training(ddp) |   ✅   |✅    |
|tensorboard callback |   ✅   |✅    |
|pretty wandb callback |  ✅  |  ❌ |
|other callbacks from pytorch_lightning |   ❌  |✅    |
|simple code |  ✅   |❌    |

```python

```

### 3, Basic Examples 

<!-- #region -->
You can follow these full examples to get started with torchkeras.

Have fun!😋😋


|example| read notebook code     |  run example in kaggle| 
|:----|:-------------------------|:-----------:|
|①kerasmodel basic 🔥🔥|  [**torchkeras.KerasModel example**](./1，kerasmodel_example.ipynb)  |  <br><div></a><a href="https://www.kaggle.com/lyhue1991/kerasmodel-example"><img src="https://kaggle.com/static/images/open-in-kaggle.svg" alt="Open In Kaggle"></a></div><br>  |
|②kerasmodel wandb 🔥🔥🔥|[**torchkeras.KerasModel with wandb demo**](./2，kerasmodel_wandb_demo.ipynb)   |  <br><div></a><a href="https://www.kaggle.com/lyhue1991/kerasmodel-wandb-example"><img src="https://kaggle.com/static/images/open-in-kaggle.svg" alt="Open In Kaggle"></a></div><br>  |
|③kerasmodel tunning 🔥🔥🔥|[**torchkeras.KerasModel with wandb sweep demo**](./3，kerasmodel_tuning_demo.ipynb)   |  <br><div></a><a href="https://www.kaggle.com/lyhue1991/torchkeras-loves-wandb-sweep"><img src="https://kaggle.com/static/images/open-in-kaggle.svg" alt="Open In Kaggle"></a></div><br>  |
|④kerasmodel tensorboard | [**torchkeras.KerasModel with tensorboard example**](./4，kerasmodel_tensorboard_demo.ipynb)   |  |
|⑤kerasmodel ddp/tpu | [**torchkeras.KerasModel  ddp tpu examples**](https://www.kaggle.com/code/lyhue1991/torchkeras-ddp-tpu-examples)   |<br><div></a><a href="https://www.kaggle.com/lyhue1991/torchkeras-ddp-tpu-examples"><img src="https://kaggle.com/static/images/open-in-kaggle.svg" alt="Open In Kaggle"></a></div><br>  |
|⑥lightmodel basic |  [**torchkeras.LightModel example**](./6，lightmodel_example.ipynb)  |   |
|⑦lightmodel tensorboard |  [**torchkeras.LightModel  with tensorboard example**](./7，lightmodel_tensorboard_demo.ipynb)  |  |

<!-- #endregion -->

**If you want to understand or modify some details of this project, feel free to read and change the source code!!!**

Any other questions, you can contact the author form the wechat official account below:

**算法美食屋** 


![](https://tva1.sinaimg.cn/large/e6c9d24egy1h41m2zugguj20k00b9q46.jpg)

