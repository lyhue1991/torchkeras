# 炼丹师，这是你的梦中情炉吗?🌹🌹


[English](README_en.md) | 简体中文


torchkeras 是一个通用的pytorch模型训练模版工具，按照如下目标进行设计和实现：

* **好看** (代码优雅，日志美丽，自带可视化)

* **好用** (使用方便，支持 进度条、评估指标、early-stopping等常用功能，支持tensorboard，wandb回调函数等扩展功能)

* **好改** (修改简单，核心代码模块化，仅约200行，并提供丰富的修改使用案例)


```python

```

## 1，炼丹之痛 😭😭


无论是学术研究还是工业落地，pytorch几乎都是目前炼丹的首选框架。

pytorch的胜出不仅在于其简洁一致的api设计，更在于其生态中丰富和强大的模型库。

但是我们会发现不同的pytorch模型库提供的训练和验证代码非常不一样。

torchvision官方提供的范例代码主要是一个关联了非常多依赖函数的train_one_epoch和evaluate函数，针对检测和分割各有一套。

yolo系列的主要是支持ddp模式的各种风格迥异的Trainer，每个不同的yolo版本都会改动很多导致不同yolo版本之间都难以通用。

抱抱脸的transformers库在借鉴了pytorch_lightning的基础上也搞了一个自己的Trainer，但与pytorch_lightning并不兼容。

非常有名的facebook的目标检测库detectron2, 也是搞了一个它自己的Trainer，配合一个全局的cfg参数设置对象来训练模型。

还有我用的比较多的语义分割的segmentation_models.pytorch这个库，设计了一个TrainEpoch和一个ValidEpoch来做训练和验证。

在学习和使用这些不同的pytorch模型库时，尝试阅读理解和改动这些训练和验证相关的代码让我受到了一万点伤害。

有些设计非常糟糕，嵌套了十几层，有些实现非常dirty，各种带下划线的私有变量满天飞。

让你每次想要改动一下加入一些自己想要的功能时就感到望而却步。

我不就想finetune一下模型嘛，何必拿这么多垃圾代码搞我？


```python

```

## 2，梦中情炉 🤗🤗

这一切的苦不由得让我怀念起tensorflow中keras的美好了。

还记得keras那compile, fit, evalute三连击吗？一切都像行云流水般自然，真正的for humans。

而且你看任何用keras实现的模型库，训练和验证都几乎可以用这一套相同的接口，没有那么多莫名奇妙的野生Trainer。

我能否基于pytorch打造一个接口和keras一样简洁易用，功能强大，但是实现代码非常简短易懂，便于修改的模型训练工具呢？

从2020年7月左右发布1.0版本到最近发布的3.86版本，我陆陆续续在工作中一边使用一边打磨一个工具，总共提交修改了70多次。

现在我感觉我细心雕琢的这个作品终于长成了我心目中接近完美的样子。


**她有一个美丽的名字：torchkeras.**
 
**是的，她兼具torch的灵动，也有keras的优雅~**

**并且她的美丽，无与伦比~**

**她，就是我的梦中情炉~ 🤗🤗**


![](./data/torchkeras.png)


```python

```


## 3，使用方法 🍊🍊


安装torchkeras
```
pip install torchkeras
```

通过使用torchkeras，你不需要写自己的pytorch模型训练循环。你只要做这样两步就可以了。

(1) 创建你的模型结构net,然后把它和损失函数传入torchkeras.KerasModel构建一个model。

(2) 使用model的fit方法在你的训练数据和验证数据上进行训练，训练数据和验证数据需要封装成两个DataLoader.



核心使用代码就像下面这样：

```python
import torch 
import torchkeras
import torchmetrics
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



## 4，主要特性 🍉🍉


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

## 5，基本范例 🌰🌰


以下范例是torchkeras的基础范例，演示了torchkeras的主要功能。

包括基础训练，使用wandb可视化，使用wandb调参，使用tensorboard可视化，使用多GPU的ddp模式训练等。


|example| notebook    |  kaggle链接| 
|:----|:-------------------------|:-----------:|
|①kerasmodel basic 🔥🔥|  [**torchkeras.KerasModel example**](./01，kerasmodel_example.ipynb)  |  <br><div></a><a href="https://www.kaggle.com/lyhue1991/kerasmodel-example"><img src="https://kaggle.com/static/images/open-in-kaggle.svg" alt="Open In Kaggle"></a></div><br>  |
|②kerasmodel wandb 🔥🔥🔥|[**torchkeras.KerasModel with wandb demo**](./02，kerasmodel_wandb_demo.ipynb)   |  <br><div></a><a href="https://www.kaggle.com/lyhue1991/kerasmodel-wandb-example"><img src="https://kaggle.com/static/images/open-in-kaggle.svg" alt="Open In Kaggle"></a></div><br>  |
|③kerasmodel tunning 🔥🔥🔥|[**torchkeras.KerasModel with wandb sweep demo**](./03，kerasmodel_tuning_demo.ipynb)   |  <br><div></a><a href="https://www.kaggle.com/lyhue1991/torchkeras-loves-wandb-sweep"><img src="https://kaggle.com/static/images/open-in-kaggle.svg" alt="Open In Kaggle"></a></div><br>  |
|④kerasmodel tensorboard | [**torchkeras.KerasModel with tensorboard example**](./04，kerasmodel_tensorboard_demo.ipynb)   |  |
|⑤kerasmodel ddp/tpu | [**torchkeras.KerasModel  ddp tpu examples**](https://www.kaggle.com/code/lyhue1991/torchkeras-ddp-tpu-examples)   |<br><div></a><a href="https://www.kaggle.com/lyhue1991/torchkeras-ddp-tpu-examples"><img src="https://kaggle.com/static/images/open-in-kaggle.svg" alt="Open In Kaggle"></a></div><br>  |

```python

```

## 6，进阶范例 🔥🔥 

下面的范例为torchkeras的进阶使用范例，由于输入数据结构的差异，这些范例有些需要对torchkeras的核心模块StepRunner进行修改。

这种修改实际上是非常简单的，保持每个模块的输出与原始实现格式一致就行，中间处理逻辑根据需要灵活调整。

这里的范例包括了使用torchkeras对一些非常常用的库中的模型进行训练的例子。

例如：

* torchvision
* transformers
* segmentation_models_pytorch

更多范例参考项目下的examples目录。

> 如果你想掌握一个东西，那么就去使用它，如果你想真正理解一个东西，那么尝试去改变它。 ———— 爱因斯坦


|example|使用模型库  |notebook |
|:----|:-----------|:-----------:|
|图片分类——Resnet|  -  | [Resnet](./examples/ResNet.ipynb) |
|语义分割——UNet|  - | [UNet](./examples/UNet.ipynb) |
|目标检测——SSD| -  | [SSD](./examples/SSD.ipynb) |
|文字识别——CRNN 🔥🔥| -  | [CRNN-CTC](./examples/CRNN_CTC.ipynb) |
|目标检测——FasterRCNN| torchvision  |  [FasterRCNN](./examples/FasterRCNN——vision.ipynb) | 
|语义分割——DeepLabV3++ 🔥| segmentation_models_pytorch |  [Deeplabv3++](./examples/Deeplabv3plus——smp.ipynb) |
|实例分割——MaskRCNN 🔥🔥| detectron2 |  [MaskRCNN](./examples/MaskRCNN——detectron2.ipynb) |
|目标检测——YOLOv8 🔥🔥| ultralytics |  [YOLOv8](./examples/YOLOv8——ultralytics.ipynb) |


```python

```

## 7，鼓励和联系作者 🎈🎈


**如果本项目对你有所帮助，想鼓励一下作者，记得给本项目加一颗星星star⭐️，并分享给你的朋友们喔😊!** 

如果在torchkeras的使用中遇到问题，可以在项目中提交issue。

如果想要获得更快的反馈或者与其他torchkeras用户小伙伴进行交流，

可以在公众号算法美食屋后台回复关键字：**加群**。

![](https://tva1.sinaimg.cn/large/e6c9d24egy1h41m2zugguj20k00b9q46.jpg)