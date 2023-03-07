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

from transformers import ViTForImageClassification, AdamW
from transformers import ViTFeatureExtractor, ViTImageProcessor
import torch.nn as nn
from torchvision.transforms import (CenterCrop, 
                                    Compose, 
                                    Normalize, 
                                    RandomHorizontalFlip,
                                    RandomResizedCrop, 
                                    Resize, 
                                    ToTensor)



# export CUDA_VISIBLE_DEVICES=1,2
# python3 train_vit_classifier.py --config ~/siamese_cities/config/ViT_classifier.yml

class ViTLightningModule(pl.LightningModule):
    def __init__(self, hparams: Namespace):
        
        self.hparams = hparams
        super(ViTLightningModule, self).__init__()
        self.vit = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224-in21k',
                                                              num_labels=self.hparams.number_cities)

        self.feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224-in21k")   

    def forward(self, pixel_values):
        outputs = self.vit(pixel_values=pixel_values)
        return outputs.logits

    def common_step(self, batch, batch_idx):
        pixel_values = batch['pixel_values']
        labels = batch['labels']
        logits = self(pixel_values)

        criterion = nn.CrossEntropyLoss()
        loss = criterion(logits, labels)
        predictions = logits.argmax(-1)
        correct = (predictions == labels).sum().item()
        accuracy = correct/pixel_values.shape[0]

        return loss, accuracy


    def training_step(self, batch, batch_idx):

        loss, accuracy = self.common_step(batch, batch_idx)  

        self.log_dict({"train_loss": loss, "train_accuracy": accuracy}, prog_bar=True, logger=True, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):

        loss, accuracy = self.common_step(batch, batch_idx)     

        self.log_dict({"val_loss": loss, "val_acc": accuracy},
                      prog_bar=True, logger=True, on_epoch=True)

        return loss


    def test_step(self, batch, batch_idx):
        loss, accuracy = self.common_step(batch, batch_idx)     

        return loss

    def configure_optimizers(self):
        return AdamW(self.parameters(), lr=5e-5)


    def train_transforms(self,image):
        

        normalize = Normalize(mean=self.feature_extractor.image_mean, std=self.feature_extractor.image_std)   

        _train_transforms = Compose(
            [
                RandomHorizontalFlip(),
                RandomResizedCrop(224, scale=(0.66, 1.0)),
                ToTensor(),
                normalize,
            ]
        )

        examples['pixel_values'] = [_train_transforms(image.convert("RGB")) for image in examples['img']]

        return examples

    def val_transforms(self,image):

        
        normalize = Normalize(mean=self.feature_extractor.image_mean, std=self.feature_extractor.image_std)

        #print(self.feature_extractor)
        _val_transforms = Compose(
            [
                Resize(256),
                CenterCrop(224),
                ToTensor(),
                normalize,
            ]
        )

        examples['pixel_values'] = [_val_transforms(image.convert("RGB")) for image in examples['img']]
        return examples

    def collate_fn(examples):
        pixel_values = torch.stack([example["pixel_values"] for example in examples])
        labels = torch.tensor([example["label"] for example in examples])
        return {"pixel_values": pixel_values, "labels": labels}

    def train_dataloader(self):

        DatasetFolder_Train = torchvision.datasets.ImageFolder(root=self.hparams.imageFolderTrain,transform=self.train_transforms)
    
        dataloader = torch.utils.data.DataLoader(
            DatasetFolder_Train,
            collate_fn=self.collate_fn,
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

        
        DatasetFolder_Valid = torchvision.datasets.ImageFolder(root=self.hparams.imageFolderValid,transform=self.val_transforms)

        dataloader = torch.utils.data.DataLoader(
            DatasetFolder_Valid,
            collate_fn=self.collate_fn,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
        )

        batch = next(iter(dataloader))

        print(batch)
        for k,v in batch.items():
            if isinstance(v, torch.Tensor):
                print(k, v.shape)
        
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

    model = ViTLightningModule(hparams=Namespace(**model_params))

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
