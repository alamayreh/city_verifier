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

# export CUDA_VISIBLE_DEVICES=0,1,2,3
# export CUDA_VISIBLE_DEVICES=4,5,6,7
# python3 train_triplet.py --config config/siamese_triplet.yml

class SiameseNetworkDatasetTriplet(Dataset):

    def __init__(self, imageFolderDataset, transform=None, num_pairs=256000):
        self.imageFolderDataset = imageFolderDataset
        self.transform = transform
        self.num_pairs = num_pairs

    def __getitem__(self, index):

        # Choose random folder
        img_a_tuple = random.choice(self.imageFolderDataset.imgs)
        
        while True:
            # keep looping till the same class image is found
            img_p_tuple = random.choice(self.imageFolderDataset.imgs)
            if ((img_a_tuple[1] == img_p_tuple[1]) and (img_a_tuple[0] != img_p_tuple[0])):
                break
        
        while True:
            # keep looping till a different class image is found
            img_n_tuple = random.choice(self.imageFolderDataset.imgs)
            if img_a_tuple[1] != img_n_tuple[1]:
                break

        img_a = Image.open(img_a_tuple[0])
        img_p = Image.open(img_p_tuple[0])
        img_n = Image.open(img_n_tuple[0])

        if self.transform is not None:
            img_a = self.transform(img_a)
            img_p = self.transform(img_p)
            img_n = self.transform(img_n)

        return img_a, img_p, img_n

    def __len__(self):
        return self.num_pairs


class SiameseNetwork(pl.LightningModule):

    def __init__(self, hparams: Namespace):
        super().__init__()

        self.hparams = hparams
        self.model, self.embedding = self.__build_model()
        self.class_weights = None
        self.criterion = torch.nn.TripletMarginLoss(margin=self.hparams.margin, swap = self.hparams.swap_triplet)
        self.total_number_training_images = 0

    def __build_model(self):
        logging.info("Build model")

        # Load resnet from torchvision
        model = models.__dict__[self.hparams.arch](
            weights='ResNet101_Weights.DEFAULT')

        nfeatures = model.fc.in_features
        model = torch.nn.Sequential(*list(model.children())[:-1])

        model.avgpool = torch.nn.AdaptiveAvgPool2d(1)
        model.flatten = torch.nn.Flatten(start_dim=1)

        # Basic embedding layer
        embedding = torch.nn.Linear(nfeatures, self.hparams.embedding_dim)

        if self.hparams.weights:
            logging.info("Load weights from pre-trained (VIPP mdoel)")
            model  = load_weights_CountryEstimation_model(model, self.hparams.weights)

        if self.hparams.freezeBackbone:
            logging.info("Freeze backbone")
            for param in model.parameters():
                param.requires_grad = False


        return model, embedding

    def forward_once(self, x):

        output = self.model(x)
        output = output.view(output.size()[0], -1)
        output = self.embedding(output)

        return output

    def forward(self, anchor, postive, negative):

        output_a = self.forward_once(anchor)
        output_p = self.forward_once(postive)
        output_n = self.forward_once(negative)

        return output_a, output_p, output_n

    def training_step(self, batch, batch_idx):

        x_a, x_p, x_n = batch

        output_a, output_p, output_n = self(x_a, x_p, x_n)
        loss = self.criterion(output_a, output_p, output_n)

        self.log("train_loss", loss, prog_bar=True, logger=True, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):

        # Calculate Triplet loss on the validation set
        x_a, x_p, x_n = batch
        
        output_a, output_p, output_n = self(x_a, x_p, x_n)
        loss = self.criterion(output_a, output_p, output_n)

        # Calculate Accuracy on the validation set:  
        # (anchor,postive -> label 0) 

        correct_a_p = 0 

        euclidean_distance_a_p = torch.nn.functional.pairwise_distance(output_a, output_p)
        pred_a_p  = torch.where(euclidean_distance_a_p > self.hparams.threshold_distance, 1, 0)
        label_a_p = torch.zeros_like(pred_a_p) 

        correct_a_p += pred_a_p.eq(label_a_p.view_as(pred_a_p)).sum().item()

        #(anchor,negatvie -> lable 1)

        correct_a_n = 0 

        euclidean_distance_a_n = torch.nn.functional.pairwise_distance(output_a, output_n)
        pred_a_n  = torch.where(euclidean_distance_a_n > self.hparams.threshold_distance, 1, 0)
        label_a_n = torch.ones_like(pred_a_n) 

        correct_a_n += pred_a_n.eq(label_a_n.view_as(pred_a_n)).sum().item()

        # Calculate accurcy 

        val_acc = 100.0 * (correct_a_n + correct_a_p) / (len(output_a) * 2 )

        self.log_dict({"val_loss": loss, "val_acc": val_acc}, prog_bar=True, logger=True, on_epoch=True)

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

        DatasetFolder_Train = torchvision.datasets.ImageFolder(
            self.hparams.imageFolderTrain)

        dataset = SiameseNetworkDatasetTriplet(
            imageFolderDataset=DatasetFolder_Train, transform=tfm_train, num_pairs = self.hparams.num_pairs)

        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
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

        DatasetFolder_Valid = torchvision.datasets.ImageFolder(
            self.hparams.imageFolderValid)

        dataset = SiameseNetworkDatasetTriplet(
            imageFolderDataset=DatasetFolder_Valid, transform=tfm_valid,num_pairs = self.hparams.num_pairs)


        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
        )

        return dataloader


def parse_args():
    args = ArgumentParser()
    args.add_argument("-c", "--config", type=Path,
                      default='config/siamese_resnet101.yml')
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
