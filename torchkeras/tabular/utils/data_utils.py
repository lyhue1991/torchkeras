import gzip
import os
import random
import shutil

import numpy as np
import pandas as pd
import requests
import torch
from pandas import DataFrame, Series
from sklearn.cluster import KMeans
from sklearn.datasets import make_classification, make_regression
from sklearn.preprocessing import LabelEncoder

def _make_smooth_weights_for_balanced_classes(y_train, mu=1.0):
    labels_dict = dict(zip(np.unique(y_train), np.bincount(y_train)))
    total = np.sum(list(labels_dict.values()))
    keys = sorted(labels_dict.keys())
    weight = []
    for i in keys:
        score = np.log(mu * total / float(labels_dict[i]))
        weight.append(score if score > 1 else 1)
    return weight


def get_class_weighted_cross_entropy(y_train, mu=1.0):
    assert y_train.ndim == 1, "Utility function only works for binary classification"
    y_train = LabelEncoder().fit_transform(y_train)
    weights = _make_smooth_weights_for_balanced_classes(y_train, mu=mu)
    criterion = torch.nn.CrossEntropyLoss(weight=torch.FloatTensor(weights))
    return criterion


def get_balanced_sampler(y_train):
    assert y_train.ndim == 1, "Utility function only works for binary classification"
    y_train = LabelEncoder().fit_transform(y_train)
    class_sample_counts = np.bincount(y_train)
    # compute weight for all the samples in the dataset
    # samples_weights contain the probability for each example in dataset to be sampled
    class_weights = 1.0 / torch.Tensor(class_sample_counts)
    train_samples_weight = [class_weights[class_id] for class_id in y_train]
    # now lets initialize samplers
    train_sampler = torch.utils.data.sampler.WeightedRandomSampler(train_samples_weight, len(y_train))
    return train_sampler


def get_gaussian_centers(y, n_components):
    if isinstance(y, Series) or isinstance(y, DataFrame):
        y = y.values
    if y.ndim == 1:
        y = y.reshape(-1, 1)
    cluster = KMeans(n_clusters=n_components, random_state=42).fit(y)
    return cluster.cluster_centers_.ravel().tolist()


def make_mixed_dataset(
    task,
    n_samples,
    n_features=7,
    n_categories=2,
    n_informative=5,
    random_state=42,
    n_targets=None,
    **kwargs,
):
    """Creates a synthetic dataset with mixed data types.

    Args:
        task (str): Either "classification" or "regression"
        n_samples (int): Number of samples to generate
        n_features (int): Number of total features to generate
        n_categories (int): Number of features to be categorical
        n_informative (int): Number of informative features
        random_state (int): Random seed for reproducibility
        n_targets (int): Number of targets to generate. n_targets>1 will generate a multi-target dataset
            for regression and multi-class dataset for classification.
            Defaults to 2 classes for classification and 1 for regression
        kwargs: Additional arguments to pass to the make_classification or make_regression function

    """
    assert n_features >= n_categories, "n_features must be greater than or equal to n_categories"
    assert n_informative <= n_features, "n_informative must be less than or equal to n_features"
    assert task in [
        "classification",
        "regression",
    ], "task must be either classification or regression"
    if n_targets is None:
        n_targets = 1 if task == "regression" else 2
    if task == "classification":
        X, y = make_classification(
            n_samples=n_samples,
            n_features=n_features,
            random_state=random_state,
            n_informative=n_informative,
            n_classes=n_targets,
            **kwargs,
        )
    elif task == "regression":
        X, y = make_regression(
            n_samples=n_samples,
            n_features=n_features,
            random_state=random_state,
            n_informative=n_informative,
            n_targets=n_targets,
            **kwargs,
        )
    cat_cols = random.choices(list(range(X.shape[-1])), k=n_categories)
    num_cols = [i for i in range(X.shape[-1]) if i not in cat_cols]
    for col in cat_cols:
        X[:, col] = pd.qcut(X[:, col], q=4).codes.astype(int)
    col_names = []
    num_col_names = []
    cat_col_names = []
    for i in range(X.shape[-1]):
        if i in cat_cols:
            col_names.append(f"cat_col_{i}")
            cat_col_names.append(f"cat_col_{i}")
        if i in num_cols:
            col_names.append(f"num_col_{i}")
            num_col_names.append(f"num_col_{i}")
    X = pd.DataFrame(X, columns=col_names)
    if n_targets == 1 or task == "classification":
        y = pd.Series(y, name="target")
    else:
        y = pd.DataFrame(y, columns=[f"target_{i}" for i in range(n_targets)])
    if task == "classification":
        y = "class_" + y.astype(str)
    data = X.join(y)
    return data, cat_col_names, num_col_names


def print_metrics(metrics, y_true, y_pred, tag, return_dict=False):
    if isinstance(y_true, pd.DataFrame) or isinstance(y_true, pd.Series):
        y_true = y_true.values
    if isinstance(y_pred, pd.DataFrame) or isinstance(y_pred, pd.Series):
        y_pred = y_pred.values
    if y_true.ndim > 1:
        y_true = y_true.ravel()
    if y_pred.ndim > 1:
        y_pred = y_pred.ravel()
    print_str_l = []
    res_d = {}
    for metric, name, params in metrics:
        score = metric(y_true, y_pred, **params)
        print_str_l.append(f"{tag} {name}: {score}")
        res_d[name] = score
    print((" | ".join(print_str_l)).strip())
    if return_dict:
        return res_d


def load_covertype_dataset(download_dir=None):
    """Predicting forest cover type from cartographic variables only (no remotely sensed data). The actual forest cover
    type for a given observation (30 x 30 meter cell) was determined from US Forest Service (USFS) Region 2 Resource
    Information System (RIS) data. Independent variables were derived from data originally obtained from US Geological
    Survey (USGS) and USFS data. Data is in raw form (not scaled) and contains binary (0 or 1) columns of data for
    qualitative independent variables (wilderness areas and soil types).

    This study area includes four wilderness areas located in the Roosevelt National Forest of northern Colorado.
    These areas represent forests with minimal human-caused disturbances, so that existing forest cover types are more a
    result of ecological processes rather than forest management practices.

    It is from [UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/covertype), but with small changes:
    - The one hot encoded columns are converted to categorical - Soli Type and Wilderness type

    Args:
        download_dir (str): Directory to download the data to. Defaults to None, which will download
            to ~/.pytorch_tabular/datasets/

    """
    if download_dir is None:
        download_dir = os.path.join(os.path.expanduser("~"), ".pytorch_tabular", "datasets")
    if not os.path.exists(download_dir):
        os.makedirs(download_dir)
    file_path = os.path.join(download_dir, "covertype.csv")
    if not os.path.exists(file_path):
        logger.info("Downloading Covertype Dataset")
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/covtype/covtype.data.gz"
        response = requests.get(url)
        with open(os.path.join(download_dir, "covertype.data.gz"), "wb") as f:
            f.write(response.content)
        with gzip.open(os.path.join(download_dir, "covertype.data.gz"), "rb") as f_in:
            with open(os.path.join(download_dir, "covertype.csv"), "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)
        os.remove(os.path.join(download_dir, "covertype.data.gz"))
    df = pd.read_csv(file_path, header=None)
    df.columns = (
        [
            "Elevation",
            "Aspect",
            "Slope",
            "Horizontal_Distance_To_Hydrology",
            "Vertical_Distance_To_Hydrology",
            "Horizontal_Distance_To_Roadways",
            "Hillshade_9am",
            "Hillshade_Noon",
            "Hillshade_3pm",
            "Horizontal_Distance_To_Fire_Points",
        ]
        + [f"Wilderness_Area_{i}" for i in range(4)]
        + [f"Soil_Type_{i}" for i in range(40)]
        + ["Cover_Type"]
    )
    # convert one hot encoded columns to categorical
    df["Wilderness_Area"] = df[[f"Wilderness_Area_{i}" for i in range(4)]].idxmax(axis=1).str.split("_").str[-1]
    df["Soil_Type"] = df[[f"Soil_Type_{i}" for i in range(40)]].idxmax(axis=1).str.split("_").str[-1]
    df.drop(
        [f"Wilderness_Area_{i}" for i in range(4)] + [f"Soil_Type_{i}" for i in range(40)],
        axis=1,
        inplace=True,
    )
    continuous_cols = [
        "Elevation",
        "Aspect",
        "Slope",
        "Horizontal_Distance_To_Hydrology",
        "Vertical_Distance_To_Hydrology",
        "Horizontal_Distance_To_Roadways",
        "Hillshade_9am",
        "Hillshade_Noon",
        "Hillshade_3pm",
        "Horizontal_Distance_To_Fire_Points",
    ]
    categorical_cols = ["Wilderness_Area", "Soil_Type"]
    return df, categorical_cols, continuous_cols, "Cover_Type"
