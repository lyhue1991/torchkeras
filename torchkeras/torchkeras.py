# -*- coding: utf-8 -*-
import os
import torch
import numpy as np
import pandas as pd
from torchkeras.summary import summary
from torchkeras.torchtools import EarlyStopping
from torchkeras.utils import log_to_message, ProgressBar

__version__ = "2.1.2"

# On macOs, run pytorch and matplotlib at the same time in jupyter should set this.
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class Model(torch.nn.Module):

    def __init__(self, net=None):
        super(Model, self).__init__()
        self.net = net

    def forward(self, x):
        if self.net:
            return self.net.forward(x)
        else:
            raise NotImplementedError

    def compile(self, loss_func, optimizer=None, metrics_dict=None, device=None):
        self.history = {}
        self.loss_func = loss_func
        self.metrics_dict = metrics_dict if metrics_dict else {}
        self.optimizer = optimizer if optimizer else torch.optim.Adam(self.parameters(), lr=0.001)
        self.device = device if torch.cuda.is_available() else None
        if self.device:
            self.to(self.device)

    def summary(self, input_shape, input_type=torch.FloatTensor, batch_size=-1):
        summary(self, input_shape, input_type, batch_size)

    def train_step(self, features, labels):

        self.train()
        self.optimizer.zero_grad()
        if self.device:
            features = features.to(self.device)
            labels = labels.to(self.device)

        # forward
        predictions = self.forward(features)
        loss = self.loss_func(predictions, labels)

        # evaluate metrics
        train_metrics = {"loss": loss.item()}
        for metrics in self.metrics_dict:
            train_metrics[metrics.__name__] = metrics(predictions, labels).item()

        # backward
        loss.backward()

        # update parameters
        self.optimizer.step()
        self.optimizer.zero_grad()

        return train_metrics

    @torch.no_grad()
    def evaluate_step(self, features, labels):

        self.eval()

        if self.device:
            features = features.to(self.device)
            labels = labels.to(self.device)

        with torch.no_grad():
            predictions = self.forward(features)
            loss = self.loss_func(predictions, labels)

        val_metrics = {"val_loss": loss.item()}
        for metrics in self.metrics_dict:
            val_metrics["val_" + metrics.__name__] = metrics(predictions, labels).item()

        return val_metrics

    def fit(self, dl_train,  dl_val=None, epochs=10, patience=10, monitor="val_loss", save_path='checkpoint.pt', verbose=True):

        dl_val = dl_val if dl_val else []
        # initialize the early_stopping object
        early_stopping = EarlyStopping(patience=patience, path=save_path, verbose=verbose)

        for epoch in range(1, epochs+1):
            print("Epoch {0} / {1}".format(epoch, epochs))
            pb = ProgressBar(len(dl_train))

            # 1，training loop -------------------------------------------------
            train_metrics_sum, step = {}, 0
            for features, labels in dl_train:
                step += 1
                train_metrics = self.train_step(features, labels)

                for name, metric in train_metrics.items():
                    train_metrics_sum[name] = train_metrics_sum.get(name, 0.0) + metric

            for name, metric_sum in train_metrics_sum.items():
                self.history[name] = self.history.get(name, []) + [metric_sum / step]

            # 2，validate loop -------------------------------------------------
            val_metrics_sum, step = {}, 0
            for features, labels in dl_val:
                step = step + 1
                val_metrics = self.evaluate_step(features, labels)
                for name, metric in val_metrics.items():
                    val_metrics_sum[name] = val_metrics_sum.get(name, 0.0) + metric
            for name, metric_sum in val_metrics_sum.items():
                self.history[name] = self.history.get(name, []) + [metric_sum / step]

            # 3，print logs -------------------------------------------------
            pb.close(log_to_message({k: round(self.history[k][-1], 4) for k in self.history}))

            # early_stopping needs the validation loss to check if it has decresed,
            # and if it has, it will make a checkpoint of the current model
            early_stopping(self.history[monitor][-1], self)
            if early_stopping.early_stop:
                print("Early stopping")
                break

        return pd.DataFrame(self.history)

    @torch.no_grad()
    def evaluate(self, dl_val):
        self.eval()
        val_metrics_list = {}
        for features, labels in dl_val:
            val_metrics = self.evaluate_step(features, labels)
            for name, metric in val_metrics.items():
                val_metrics_list[name] = val_metrics_list.get(name, []) + [metric]

        return {name: np.mean(metric_list) for name, metric_list in val_metrics_list.items()}

    @torch.no_grad()
    def predict(self, dl):
        self.eval()
        if self.device:
            result = torch.cat([self.forward(t[0].to(self.device)) for t in dl])
        else:
            result = torch.cat([self.forward(t[0]) for t in dl])
        return result.data
