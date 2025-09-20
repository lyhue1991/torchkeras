import torch 
from torch.utils.data import Dataset 
import numpy as np 
import pandas as pd 
from typing import Dict, Iterable, List, Optional, Tuple, Union

class TabularDataset(Dataset):
    def __init__(
        self,
        data: pd.DataFrame,
        task: str,
        continuous_cols: List[str] = None,
        categorical_cols: List[str] = None,
        target: List[str] = None,
        start_id: int = 0,
    ):
        """Dataset to Load Tabular Data.

        Args:
            data (pd.DataFrame): Pandas DataFrame to load during training
            task (str):
                It is a regression/binary/multilass  task. If multilass, it returns a LongTensor as target
            continuous_cols (List[str], optional): A list of names of continuous columns. Defaults to None.
            categorical_cols (List[str], optional): A list of names of categorical columns.
                These columns must be ordinal encoded beforehand. Defaults to None.
            target (List[str], optional): A list of strings with target column name(s). Defaults to None.

        """

        self.task = task
        self.n = data.shape[0]
        self.target = target
        self.start_id = start_id 
        self.ids = np.arange(start_id,start_id+self.n)
        if target:
            self.y = data[target].astype(np.float32).values
            if isinstance(target, str):
                self.y = self.y.reshape(-1, 1)  
        else:
            self.y = np.zeros((self.n, 1))  

        if task in ("multiclass","classification"):
            self.y = self.y.astype(np.int64)
   
        self.categorical_cols = categorical_cols if categorical_cols else []
        self.continuous_cols = continuous_cols if continuous_cols else []

        if self.continuous_cols:
            self.continuous_X = data[self.continuous_cols].astype(np.float32).values

        if self.categorical_cols:
            self.categorical_X = data[categorical_cols]
            self.categorical_X = self.categorical_X.astype(np.int64).values

    @property
    def data(self):
        """Returns the data as a pandas dataframe."""
        if self.continuous_cols and self.categorical_cols:
            data = pd.DataFrame(
                np.concatenate([self.categorical_X, self.continuous_X], axis=1),
                columns=self.categorical_cols + self.continuous_cols,
            )
        elif self.continuous_cols:
            data = pd.DataFrame(self.continuous_X, columns=self.continuous_cols)
        elif self.categorical_cols:
            data = pd.DataFrame(self.categorical_X, columns=self.categorical_cols)
        else:
            data = pd.DataFrame()
        for i, t in enumerate(self.target):
            data[t] = self.y[:, i]
        return data

    def __len__(self):
        """Denotes the total number of samples."""
        return self.n

    def __getitem__(self, idx):
        """Generates one sample of data."""
        return {
            "target": torch.tensor(self.y[idx]),
            "continuous": (torch.tensor(self.continuous_X[idx]) if self.continuous_cols else torch.Tensor()),
            "categorical": (torch.tensor(self.categorical_X[idx]) if self.categorical_cols else torch.Tensor()),
            "id": torch.tensor(self.ids[idx],dtype=torch.long),
        }
    def get_batch(self, batch_ids):
        batch_idxes = batch_ids - self.start_id
        batch = {
            "target": torch.tensor(self.y[batch_idxes]),
            "continuous": (torch.tensor(self.continuous_X[batch_idxes]) if self.continuous_cols else torch.Tensor()),
            "categorical": (torch.tensor(self.categorical_X[batch_idxes]) if self.categorical_cols else torch.Tensor()),
            "id": torch.tensor(self.ids[batch_idxes],dtype=torch.long),
        }
        return batch 
        

