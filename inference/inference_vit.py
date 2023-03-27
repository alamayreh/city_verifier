import os
import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import sys
sys.path.insert(0, '/data/omran/siamese_cities')
import torch
import logging
import torchvision
import pandas as pd
from PIL import Image
from argparse import  ArgumentParser
from pathlib import Path
from utils import *
from training.train_classifier import SiameseNetwork as Classifier
from tqdm import tqdm
from os import listdir
from os.path import isfile, join
import math
from scipy import spatial
import random


random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

import os
import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


os.environ["CUDA_VISIBLE_DEVICES"] = "7"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dataset directories
TEST_DATASET_DIR = '/data/omran/cities_data/dataset/filtered/dataset_10k/test'
NUM_CLASSES = 10



def create_dataset(batch_size=32):

    test_transform = transforms.Compose([transforms.Resize(256),
                                         transforms.CenterCrop(224),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    test_dataset = datasets.ImageFolder(TEST_DATASET_DIR, transform=test_transform)

    print(f"Found {len(test_dataset)} test images")

    return DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


def evaluate_model(loader, model, optimizer, criterion):

    model.eval()

    avg_loss_batch, avg_acc_batch, count_batches = 0.0, 0, 0
    for inputs, labels in loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        with torch.set_grad_enabled(False):
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

        avg_loss_batch += loss.item()
        avg_acc_batch += torch.sum(preds.eq(labels)).item() / len(inputs)
        count_batches += 1

    test_loss = avg_loss_batch / count_batches
    test_acc = avg_acc_batch / count_batches

    return test_loss, test_acc


def load_model(model_path, n_classes):

    model = torchvision.models.vit_l_16(weights=torchvision.models.ViT_L_16_Weights.DEFAULT).to(device)
    last_layer_features = model.heads.head.in_features
    model.heads.head = torch.nn.Linear(in_features=last_layer_features, out_features=NUM_CLASSES, bias=True)
    model.num_classes = n_classes

    sd = torch.load(model_path)
    print(
        f'Loaded best model obtained after {sd["epoch"]} epochs with validation accuracy {sd["acc"]:.3f} and loss {sd["loss"]:.3f}')

    pretrained_state_dict = {k.replace('module.', ''): v for k, v in sd["model_state_dict"].items()}
    model.load_state_dict(pretrained_state_dict)

    return model.to(device)


if __name__ == '__main__':

    test_dataloader = create_dataset(batch_size=256)

    trained_model = load_model('/data/omran/cities_data/models/pretrain_vit/vit_16h_model.pt', NUM_CLASSES)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(trained_model.parameters(), lr=0.01, momentum=0.9)

    # Test
    test_loss, test_accuracy = evaluate_model(loader=test_dataloader,
                                              model=trained_model,
                                              criterion=criterion,
                                              optimizer=optimizer)

    print(f"Test loss: {test_accuracy}")
    print(f"Test accuracy: {test_loss}")