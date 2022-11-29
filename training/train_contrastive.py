import random
import torch
import logging
import torchvision
import torch.nn.functional as F
from PIL import Image
import pytorch_lightning as pl
import numpy as np
from argparse import Namespace, ArgumentParser
import torchvision.models as models
from pathlib import Path
import os.path
import yaml
from datetime import datetime
from utils import *
import torchvision.models.detection as detection

# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# python3 train.py --config config/siamese_resnet101.yml


class ContrastiveLoss(torch.nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """

    def __init__(self, margin):
        super().__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(
            output1, output2, keepdim=True)
        loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) + (
            label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))

        return loss_contrastive


class SiameseNetworkDataset(Dataset):

    def __init__(self, imageFolderDataset, transform=None,num_pairs=25600):
        self.imageFolderDataset = imageFolderDataset
        self.transform = transform
        self.num_pairs = num_pairs

    def __getitem__(self, index):
        img0_tuple = random.choice(self.imageFolderDataset.imgs)

        # print(self.imageFolderDataset.img)
        # we need to make sure approx 50% of images are in the same class
        should_get_same_class = random.randint(0, 1)
        # print("should_get_same_class",should_get_same_class)
        if should_get_same_class:
            while True:
                # keep looping till the same class image is found
                img1_tuple = random.choice(self.imageFolderDataset.imgs)
                # print('img0_tuple',img0_tuple)
                # print('img1_tuple',img1_tuple)
                if ((img0_tuple[1] == img1_tuple[1]) and (img0_tuple[0] != img1_tuple[0])):
                    # print("nowBreak")
                    break
        else:
            while True:
                # keep looping till a different class image is found

                img1_tuple = random.choice(self.imageFolderDataset.imgs)
                # print('img0_tuple',img0_tuple)
                # print('img1_tuple',img1_tuple)
                if img0_tuple[1] != img1_tuple[1]:
                    # print("nowBreak")
                    break

        img0 = Image.open(img0_tuple[0])
        img1 = Image.open(img1_tuple[0])
        #img0 = img0.convert("L")
        #img1 = img1.convert("L")

        # if self.should_invert:
        #    img0 = PIL.ImageOps.invert(img0)
        #    img1 = PIL.ImageOps.invert(img1)

        if self.transform is not None:
            img0 = self.transform(img0)
            img1 = self.transform(img1)

        return img0, img1, torch.from_numpy(np.array([int(img1_tuple[1] != img0_tuple[1])], dtype=np.float32))

    def __len__(self):
        #return len(self.imageFolderDataset.imgs)
        return self.num_pairs


class SiameseNetwork(pl.LightningModule):

    def __init__(self, hparams: Namespace):
        super().__init__()

        self.hparams = hparams
        self.model, self.embedding = self.__build_model()
        self.class_weights = None
        self.criterion = ContrastiveLoss(margin=self.hparams.margin)
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

        # Basic embedding layers
        embedding = torch.nn.Linear(nfeatures, self.hparams.embedding_dim)

        if self.hparams.weights:
            logging.info("Load weights from pre-trained model")
            model, embedding = load_weights_if_available(
                model, embedding, self.hparams.weights
            )
        

        if self.hparams.freezeBackbone:

            logging.info("Freeze backbone")
            for param in model.parameters():
                param.requires_grad = False
            #for child in model.children():
            #    for param in child.parameters():
            #        param.requires_grad = False

        return model, embedding

    def forward_once(self, x):
        output = self.model(x)
        #print("\noutput befor view : ", output.size())
        output = output.view(output.size()[0], -1)
        #print("output after view : ", output.size())
        output = self.embedding(output)
        #print("output after embedding : ", output.size())
        return output

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        return output1, output2

    def training_step(self, batch, batch_idx):
        x0, x1, y = batch
        output1, output2 = self(x0, x1)
        loss = self.criterion(output1, output2, y)

        self.log("train_loss", loss, prog_bar=True, logger=True,on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x0, x1, target = batch
        output_a, output_n = self(x0, x1)
        loss = self.criterion(output_a, output_n, target)

        correct = 0

        euclidean_distance_a_n = torch.nn.functional.pairwise_distance(
            output_a, output_n)

        pred = torch.where(euclidean_distance_a_n >
                           self.hparams.threshold_distance, 1, 0)
        correct += pred.eq(target.view_as(pred)).sum().item()

        val_acc = 100. * correct / len(output_a)

        # Calculate accurcy

        self.log_dict({"val_loss": loss, "val_acc": val_acc},
                      prog_bar=True, logger=True, on_epoch=True)

        return loss

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def configure_optimizers(self):

        if (self.hparams.optim["type"] == "SGD"):
            optim_feature_extrator = torch.optim.SGD(
                self.parameters(), **self.hparams.optim["params"]
            )
            

        elif (self.hparams.optim["type"] == "Adam"):
            optim_feature_extrator = torch.optim.Adam(
                self.parameters() #, **self.hparams.optim["params"]
            )

        else:
            raise ValueError("Provide valid optimizer type.")

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

        # logging.info("train_dataloader")
        # Training images transform

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

        dataset = SiameseNetworkDataset(
            imageFolderDataset=DatasetFolder_Train, transform=tfm_train, num_pairs = self.hparams.num_pairs)

        dataloader = torch.utils.data.DataLoader(
            dataset,
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

        # logging.info("val_dataloader")

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

        dataset = SiameseNetworkDataset(
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
                      default='config/siamese_resnet101_contrastive.yml')
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
