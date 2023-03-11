
# Pytorch❤️Keras

[English](README_en.md) | 简体中文


torchkeras 是一个通用的pytorch模型训练模版工具，按照如下目标进行设计和实现：

* **好看** (代码优雅，日志美丽，自带可视化)

* **好用** (使用方便，支持 进度条、评估指标、early-stopping等常用功能，支持tensorboard，wandb回调函数等扩展功能)

* **好改** (修改简单，核心代码模块化，仅约200行，并提供丰富的修改使用案例)




## 1, 使用方法



通过使用torchkeras，你不需要写自己的pytorch模型训练循环。你只要做这样两步就可以了。

(1) 创建你的模型结构net,然后把它和损失函数传入torchkeras.KerasModel构建一个model。

(2) 使用model的fit方法在你的训练数据和验证数据上进行训练，训练数据和验证数据需要封装成两个DataLoader.


<!-- #region -->
核心使用代码就像下面这样：

```python
import torch 
import torchkeras

model = torchkeras.KerasModel(net,
                              loss_fn = nn.BCEWithLogitsLoss(),
                              optimizer= torch.optim.Adam(net.parameters(),lr = 1e-4),
                              metrics_dict = {"acc":torchmetrics.Accuracy(task='binary')}
                             )
dfhistory=model.fit(train_data=dl_train, 
                    val_data=dl_val, 
                    epochs=20, 
                    patience=3, 
                    ckpt_path='checkpoint.pt',
                    monitor="val_acc",
                    mode="max",
                    plot=True
                   )

```

在jupyter notebook中执行训练代码，你将看到类似下面的训练可视化图像和训练日志进度条。

![](./data/train_fig.png)

![](./data/train_log.png)
<!-- #endregion -->

```python

```

## 2,  主要特性


torchkeras 支持以下这些功能特性，稳定支持这些功能的起始版本以及这些功能借鉴或者依赖的库的来源见下表。




|功能| 稳定支持起始版本 | 依赖或借鉴库 |
|:----|:-------------------:|:--------------|
|✅ 训练进度条 | 3.0.0   | 依赖tqdm,借鉴keras|
|✅ 训练评估指标  | 3.0.0   | 借鉴pytorch_lightning |
|✅ notebook中训练自带可视化 |  3.8.0  |借鉴fastai |
|✅ early stopping | 3.0.0   | 借鉴keras |
|✅ gpu training | 3.0.0    |依赖accelerate|
|✅ multi-gpus training(ddp) |   3.6.0 | 依赖accelerate|
|✅ fp16/bf16 training|   3.6.0  | 依赖accelerate|
|✅ tensorboard callback |   3.7.0  |依赖tensorboard |
|✅ wandb callback |  3.7.0 |依赖wandb |

```python

```

## 3, 基本范例

<!-- #region -->
以下范例是torchkeras的基础范例，演示了torchkeras的主要功能。

包括基础训练，使用wandb可视化，使用wandb调参，使用tensorboard可视化，使用多GPU的ddp模式训练等。


|example| notebook    |  kaggle链接| 
|:----|:-------------------------|:-----------:|
|①kerasmodel basic 🔥🔥|  [**torchkeras.KerasModel example**](./1，kerasmodel_example.ipynb)  |  <br><div></a><a href="https://www.kaggle.com/lyhue1991/kerasmodel-example"><img src="https://kaggle.com/static/images/open-in-kaggle.svg" alt="Open In Kaggle"></a></div><br>  |
|②kerasmodel wandb 🔥🔥🔥|[**torchkeras.KerasModel with wandb demo**](./2，kerasmodel_wandb_demo.ipynb)   |  <br><div></a><a href="https://www.kaggle.com/lyhue1991/kerasmodel-wandb-example"><img src="https://kaggle.com/static/images/open-in-kaggle.svg" alt="Open In Kaggle"></a></div><br>  |
|③kerasmodel tunning 🔥🔥🔥|[**torchkeras.KerasModel with wandb sweep demo**](./3，kerasmodel_tuning_demo.ipynb)   |  <br><div></a><a href="https://www.kaggle.com/lyhue1991/torchkeras-loves-wandb-sweep"><img src="https://kaggle.com/static/images/open-in-kaggle.svg" alt="Open In Kaggle"></a></div><br>  |
|④kerasmodel tensorboard | [**torchkeras.KerasModel with tensorboard example**](./4，kerasmodel_tensorboard_demo.ipynb)   |  |
|⑤kerasmodel ddp/tpu | [**torchkeras.KerasModel  ddp tpu examples**](https://www.kaggle.com/code/lyhue1991/torchkeras-ddp-tpu-examples)   |<br><div></a><a href="https://www.kaggle.com/lyhue1991/torchkeras-ddp-tpu-examples"><img src="https://kaggle.com/static/images/open-in-kaggle.svg" alt="Open In Kaggle"></a></div><br>  |
<!-- #endregion -->

```python

```

## 4, 进阶范例

<!-- #region -->
下面的范例为torchkeras的进阶使用范例，由于输入数据结构的差异，这些范例有些需要对torchkeras的核心模块StepRunner进行修改。

这种修改实际上是非常简单的，保持每个模块的输出与原始实现格式一致就行，中间处理逻辑根据需要灵活调整。


> 如果你想掌握一个东西，那么就去使用它，如果你想真正理解一个东西，那么尝试去改变它。 ———— 爱因斯坦




<!-- #endregion -->


|example|  kaggle链接| 
|:----|:-----------:|
|图片分类——Resnet|    <br><div></a><a href="https://www.kaggle.com/code/lyhue1991/torchkeras-resnet"><img src="https://kaggle.com/static/images/open-in-kaggle.svg" alt="Open In Kaggle"></a></div><br>  |
|语义分割——UNet|   <br><div></a><a href="https://www.kaggle.com/code/lyhue1991/torchkeras-unet"><img src="https://kaggle.com/static/images/open-in-kaggle.svg" alt="Open In Kaggle"></a></div><br>  |
|目标检测——SSD|   <br><div></a><a href="https://www.kaggle.com/lyhue1991/torchkeras-ssd"><img src="https://kaggle.com/static/images/open-in-kaggle.svg" alt="Open In Kaggle"></a></div><br>  |



使用过程中遇到问题推荐优先阅读范例和源码，确实无法解决的可以在公众号算法美食屋留言。



![](https://tva1.sinaimg.cn/large/e6c9d24egy1h41m2zugguj20k00b9q46.jpg)

