import sys
sys.path.insert(0, '/data/omran/siamese_cities')
import random
import torch
import logging
import torchvision
from PIL import Image
import pytorch_lightning as pl
import numpy as np
from argparse import Namespace, ArgumentParser
import torchvision.models as models
from pathlib import Path
import yaml
from datetime import datetime
from utils import *
import math
from scipy import spatial
import pandas as pd
import os
from os.path import isfile, join

# export CUDA_VISIBLE_DEVICES=4,5,6,7
# python3 train_classifier.py --config ~/siamese_cities/config/cities_classifier.yml

class SiameseNetwork(pl.LightningModule):

    def __init__(self, hparams: Namespace):
        super().__init__()

        self.hparams = hparams
        self.model,  self.embedding_fc  = self.__build_model()
        self.class_weights = None
        self.total_number_training_images = 0 # just to print the total number of images only in the first loop

    def __build_model(self):
        logging.info("Build model")

        # Load resnet from torchvision
        #model = models.__dict__[self.hparams.arch](
        #    weights='ResNet18_Weights.DEFAULT')

        model = models.__dict__[self.hparams.arch](
            weights='ResNet101_Weights.DEFAULT')

        nfeatures = model.fc.in_features
        model = torch.nn.Sequential(*list(model.children())[:-1])
        model.avgpool = torch.nn.AdaptiveAvgPool2d(1)
        model.flatten = torch.nn.Flatten(start_dim=1)

        # Basic embedding layers
        embedding_fc = torch.nn.Linear(nfeatures, self.hparams.number_cities)

        if self.hparams.weights:
            logging.info("Load weights from pre-trained (VIPP mdoel)")
            model = load_weights_CountryEstimation_model(
                model, self.hparams.weights
            )


        return model, embedding_fc


    def forward(self, x):

        output = self.model(x)
        output = output.view(output.size()[0], -1)

        yhats = self.embedding_fc(output)

        return yhats

    def training_step(self, batch, batch_idx):

        images, target = batch

        output = self(images)

       
        loss = torch.nn.functional.cross_entropy(output, target)

        self.log("train_loss", loss, prog_bar=True, logger=True, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):

        #logging.info("start validation step")

        images, target = batch

        output = self(images)

        #print(x0.size())
        #print(similarity.size())
        #print(target.size())

        loss = torch.nn.functional.cross_entropy(output, target)

        #loss = self.criterion(output_model, target)

        correct = 0

        probs = torch.softmax(output, dim=1)
        #print(f'probs : {probs}')
        pred = probs.argmax(dim=1)

        #print(f'pred : {pred}')
        #print(f'target : {target}')
        #pred = torch.where(output_model > 0.5, 1, 0)

        correct += pred.eq(target.view_as(pred)).sum().item()
        #print(f'correct : {correct}')
        val_acc = 100. * correct / len(output)

        self.log_dict({"val_loss": loss, "val_acc": val_acc},
                      prog_bar=True, logger=True, on_epoch=True)

        #logging.info("End validation step")

        return loss


    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def configure_optimizers(self):

        optim_feature_extrator = torch.optim.SGD(
            self.parameters(), **self.hparams.optim["params"]
        )

        return {
            "optimizer": optim_feature_extrator,
            "lr_scheduler": {
                "scheduler": torch.optim.lr_scheduler.MultiStepLR(
                    optim_feature_extrator, **self.hparams.scheduler["params"]
                ),
                "interval": "epoch",
                "name": "lr",
            },
        }

    def train_dataloader(self):

        tfm_train = torchvision.transforms.Compose(
            [
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.RandomResizedCrop(
                    224, scale=(0.66, 1.0)),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(
                    (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
                ),
            ]
        )
        DatasetFolder_Train = torchvision.datasets.ImageFolder(root=self.hparams.imageFolderTrain,transform=tfm_train)
    
        dataloader = torch.utils.data.DataLoader(
            DatasetFolder_Train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=True,
            pin_memory=True,
        )

        if (self.total_number_training_images == 0):

            # To print the number of training images
            self.total_number_training_images = len(dataloader.dataset)
            logging.info(
                f"\nThe total number of samples : {self.total_number_training_images}")

        return dataloader

    def val_dataloader(self):

        tfm_valid = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize(256),
                torchvision.transforms.CenterCrop(224),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(
                    (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
                ),
            ]
        )

        DatasetFolder_Valid = torchvision.datasets.ImageFolder(root=self.hparams.imageFolderValid,transform=tfm_valid)

        dataloader = torch.utils.data.DataLoader(
            DatasetFolder_Valid,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
        )

        return dataloader


def parse_args():
    args = ArgumentParser()
    args.add_argument("-c", "--config", type=Path)
    args.add_argument("--progbar", action="store_true")
    return args.parse_args()


if __name__ == '__main__':

    args = parse_args()
    logging.basicConfig(level=logging.INFO)

    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    model_params = config["model_params"]
    trainer_params = config["trainer_params"]

    out_dir = Path(config["out_dir"]) / datetime.now().strftime("%y%m%d-%H%M")
    out_dir.mkdir(exist_ok=True, parents=True)
    logging.info(f"Output directory: {out_dir}")

    model = SiameseNetwork(hparams=Namespace(**model_params))

    logger = pl.loggers.TensorBoardLogger(
        save_dir=str(out_dir), name="tb_logs")
    checkpoint_dir = out_dir / "ckpts" / "{epoch:03d}-{val_loss:.4f}"
    #checkpoint_dir = out_dir / "ckpts"
    checkpointer = pl.callbacks.model_checkpoint.ModelCheckpoint(
        checkpoint_dir)

    trainer = pl.Trainer(
        **trainer_params,
        logger=logger,
        val_check_interval=model_params["val_check_interval"],
        checkpoint_callback=checkpointer,
        progress_bar_refresh_rate=1,
    )

    trainer.fit(model)
