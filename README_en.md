
# Pytorch❤️Keras

English | [简体中文](README.md)


The torchkeras library is a simple tool for training neural network in pytorch jusk in a keras style. 😋😋


## 1, Introduction


With torchkeras, You need not to write your training loop with many lines of code, all you need to do is just 

like these two steps as below:

(i) create your network and wrap it and the loss_fn together with torchkeras.KerasModel like this: 
`model = torchkeras.KerasModel(net,loss_fn=nn.BCEWithLogitsLoss())`.

(ii) fit your model with the training data and validate data.


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
                    mode="max",
                    plot = True
                   )

```

![](./data/train_fig.png)

![](./data/train_log.png)


<!-- #endregion -->

**This project seems somehow powerful, but the source code is very simple.**

**Actually, only about 200 lines of Python code.**

**If you want to understand or modify some details of this project, feel free to read and change the source code!!!**

```python

```

## 2, Features 


The main features supported by torchkeras are listed below.

Versions when these features are introduced and the libraries which they used  or inspired from are given.



|features| supported from version | used or inspired by library  |
|:----|:-------------------:|:--------------|
|✅ training progress bar | 3.0.0   | use tqdm,inspired by keras|
|✅ training metrics  | 3.0.0   | inspired by pytorch_lightning |
|✅ notebook visualization in traning |  3.8.0  |inspired by fastai |
|✅ early stopping | 3.0.0   | inspired by keras |
|✅ gpu training | 3.0.0    |use accelerate|
|✅ multi-gpus training(ddp) |   3.6.0 | use accelerate|
|✅ fp16/bf16 training|   3.6.0  | use accelerate|
|✅ tensorboard callback |   3.7.0  |use tensorboard |
|✅ wandb callback |  3.7.0 |use wandb |


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

<!-- #endregion -->

**If you want to understand or modify some details of this project, feel free to read and change the source code!!!**

Any other questions, you can contact the author form the wechat official account below:

**算法美食屋** 


![](https://tva1.sinaimg.cn/large/e6c9d24egy1h41m2zugguj20k00b9q46.jpg)

