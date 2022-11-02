import torch
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import torchvision
import torch.nn.functional as F
import numpy as np
import logging
from collections import OrderedDict
from pathlib import Path
from typing import Union, List


def load_weights_if_available(
    model: torch.nn.Module, embedding: torch.nn.Module, weights_path: Union[str, Path]
):

    checkpoint = torch.load(weights_path, map_location=lambda storage, loc: storage)

    state_dict_features = OrderedDict()
    state_dict_embedding = OrderedDict()
    for k, w in checkpoint["state_dict"].items():
        if k.startswith("model"):
            state_dict_features[k.replace("model.", "")] = w
        elif k.startswith("embedding"):
            state_dict_embedding[k.replace("embedding.", "")] = w
        else:
            logging.warning(f"Unexpected prefix in state_dict due to loading from Country Estimation model: {k}")
    model.load_state_dict(state_dict_features, strict=True)
    return model, embedding


def load_weights_CountryEstimation_model(
    model: torch.nn.Module, weights_path: Union[str, Path]
):

    checkpoint = torch.load(weights_path, map_location=lambda storage, loc: storage)

    state_dict_features = OrderedDict()
    state_dict_embedding = OrderedDict()
    for k, w in checkpoint["state_dict"].items():
        if k.startswith("model"):
            state_dict_features[k.replace("model.", "")] = w
        elif k.startswith("embedding"):
            state_dict_embedding[k.replace("embedding.", "")] = w
        else:
            logging.warning(f"Unexpected prefix in state_dict: {k}")
    model.load_state_dict(state_dict_features, strict=True)
    return model


def load_weights_Sigmoid_model(
    model: torch.nn.Module, weights_path: Union[str, Path]
):

    checkpoint = torch.load(weights_path, map_location=lambda storage, loc: storage)

    state_dict_features = OrderedDict()
    state_dict_embedding = OrderedDict()
    for k, w in checkpoint["state_dict"].items():
        if k.startswith("model"):
            state_dict_features[k.replace("model.", "")] = w
        elif k.startswith("embedding"):
            state_dict_embedding[k.replace("embedding.", "")] = w
        else:
            logging.warning(f"Unexpected prefix in state_dict: {k}")
    model.load_state_dict(state_dict_features, strict=True)
    return model