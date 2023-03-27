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
# python3 train_vit_siamese.py --config ~/siamese_cities/config/siamese_vit.yml

class SiameseNetworkDataset(Dataset):

    def __init__(self, imageFolderDataset, database_csv_File,database_vipp_file,similarity_training, transform=None, num_pairs=None):

        self.imageFolderDataset = imageFolderDataset
        self.transform = transform
        self.num_pairs = num_pairs
        self.similarity_training = similarity_training
        self.database_csv = pd.read_csv(database_csv_File,usecols=['IMG_ID','S16']).set_index('IMG_ID')
        self.database_vipp = pd.read_csv(database_vipp_file,usecols=['IMG_ID','pred_10_classes']).set_index('IMG_ID')
        self.vipp_classes = {"Amsterdam":9,"Barcelona":2,"Berlin":4,"Paris":3,"Cairo":44,"Delhi":14,"London":1,"Edinburgh":1,"Moscow":22,"St_Petersburg":22,"NewYork":0,"LosAngeles":0,"Rio_de_Janeiro":11,"Rome":5,"Milan":5,"Shanghai":10,"Beijing":10,"Sydney":8,"Tokyo":7}   
        #self.vipp_classes = {"Munich":4,"Berlin":4,"Cairo":44,"Delhi":14,"London":1,"Edinburgh":1,"Moscow":22,"St_Petersburg":22,"New_york":0,"Rio_de_Janeiro":11,"Roma":5,"Milan":5,"Shanghai":10,"Beijing":10,"Tokyo":7}   
 
    def string_to_prob(self, string_prob):

        # Read probability from datafram
        image_prob_str = ((string_prob)[1:])[:-1].split()
        image_prob = [float(i) for i in image_prob_str]

        return image_prob
  
    def distance_euclidean(self, prob_0, prob_1):

        prob_0 = self.string_to_prob(prob_0)
        prob_1 = self.string_to_prob(prob_1)

        eDistance = math.dist((prob_0), (prob_1))

        return (eDistance)

    def distance_cos(self, prob_0, prob_1):

        prob_0 = self.string_to_prob(prob_0)
        prob_1 = self.string_to_prob(prob_1)

        cDistance = spatial.distance.cosine(prob_0, prob_1)

        return (1- cDistance)

    def get_IMG_ID(self,path_string):
        image_prob_str = path_string.split('/')
        return image_prob_str[-1]

    def get_vipp_class(self,path_string):
        image_prob_str = path_string.split('/')
        return int (self.vipp_classes[image_prob_str[-2]])

    def get_top(self,top10,num):
            
        image_prob_str = ((top10.strip())[1:])[:-1].split()
        list10 = [int(i) for i in image_prob_str]

        return list10[0:num]  

    def __getitem__(self, index):

        img0_tuple = random.choice(self.imageFolderDataset.imgs)

        # we need to make sure approx 50% of images are in the same class
        should_get_same_class = random.randint(0, 1)
        cos_dist = 0

        if(self.similarity_training):
            if should_get_same_class:
                while True:
                    # keep looping till the same class image is found
                    img1_tuple = random.choice(self.imageFolderDataset.imgs)
                    if ((img0_tuple[1] == img1_tuple[1]) and (img0_tuple[0] != img1_tuple[0])):
                        img0_ID = (self.get_IMG_ID(img0_tuple[0]))
                        img1_ID = (self.get_IMG_ID(img1_tuple[0]))
                        cos_dist = float(self.distance_cos(self.database_csv.loc[img0_ID].S16,self.database_csv.loc[img1_ID].S16)) 
                        if(cos_dist>0.5):
                            break
            else:
                while True:
                    # keep looping till a different class image is found
                    img1_tuple = random.choice(self.imageFolderDataset.imgs)

                    if img0_tuple[1] != img1_tuple[1]:
                        img0_ID = (self.get_IMG_ID(img0_tuple[0]))
                        img1_ID = (self.get_IMG_ID(img1_tuple[0]))
                        cos_dist = float(self.distance_cos(self.database_csv.loc[img0_ID].S16,self.database_csv.loc[img1_ID].S16)) 
                        if(cos_dist>0.5):
                            break                    
                      
        else:

            if should_get_same_class:
                while True:
                    # keep looping till the same class image is found
                    img1_tuple = random.choice(self.imageFolderDataset.imgs)
                    
                    img0_ID = (self.get_IMG_ID(img0_tuple[0]))
                    img1_ID = (self.get_IMG_ID(img1_tuple[0]))                                      
                    
                    if ((img0_tuple[1] == img1_tuple[1]) and (img0_tuple[0] != img1_tuple[0])):
                        break
                    
            else:
                while True:
                    # keep looping till a different class image is found
                    img1_tuple = random.choice(self.imageFolderDataset.imgs)

                    img0_ID = (self.get_IMG_ID(img0_tuple[0]))
                    img1_ID = (self.get_IMG_ID(img1_tuple[0]))

                    img0_class = (self.get_vipp_class(img0_tuple[0]))
                    #img1_class = (self.get_vipp_class(img0_tuple[0]))

                    
                    if (img0_tuple[1] != img1_tuple[1] and img0_class in self.get_top(self.database_vipp.loc[img1_ID].pred_10_classes,num=3)):
                        break                    
                
        img0 = Image.open(img0_tuple[0])
        img1 = Image.open(img1_tuple[0])

        if self.transform is not None:
            img0 = self.transform(img0)
            img1 = self.transform(img1)

        
        #print(f'{img0_tuple[0]} | {img1_tuple[0]} \n Simialrity {cos_dist}')
        #print('------------------------------------------------------------------------------------------------------')    
        
        #print(f'{img0_tuple[0]} | {img1_tuple[0]} \n img0_class : { (self.get_vipp_class(img0_tuple[0]))} img1_top3 : {self.get_top(self.database_vipp.loc[img1_ID].pred_10_classes,num=3)} pair {int(img1_tuple[1] != img0_tuple[1])}')
        #print('------------------------------------------------------------------------------------------------------')    

        # Same 0 city, diff 1 citiy     
        similarity = 1  
        return img0, img1, torch.from_numpy(np.array([int(img1_tuple[1] != img0_tuple[1])], dtype=np.float32)),torch.from_numpy(np.array([similarity],dtype=np.float32))

    def __len__(self):
        # return len(self.imageFolderDataset.imgs)
        return self.num_pairs


class SiameseNetwork(pl.LightningModule):

    def __init__(self, hparams: Namespace):
        super().__init__()

        self.hparams = hparams
        self.model,  self.embedding, self.sigmoid = self.__build_model()
        self.class_weights = None
        self.total_number_training_images = 0 # just to print the total number of images only in the first loop

    def __build_model(self):
        logging.info("Build model")

        # Load resnet from torchvision
        model = torchvision.models.vit_l_16(weights=torchvision.models.ViT_L_16_Weights.DEFAULT)
        
        # set the last layer to embedding dim
        last_layer_features = model.heads.head.in_features

        model.heads.head = torch.nn.Linear(in_features=last_layer_features, out_features= self.hparams.embedding_dim, bias=True) 

        # print the last layer 
        #print(list(model.children())[-1])
        #print(model)

        embedding = torch.nn.Sequential(
            torch.nn.Linear(self.hparams.embedding_dim *
                            2, self.hparams.embedding_dim, bias=True),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(self.hparams.embedding_dim, 1),
        )

        if self.hparams.weights:
            logging.info("Load weights from pre-trained (ViT andrea)")
            sd = torch.load(self.hparams.weights)
            print(f'Loaded best model obtained after {sd["epoch"]} epochs with validation accuracy {sd["acc"]:.3f} and loss {sd["loss"]:.3f}')
            pretrained_state_dict = {k.replace('module.', ''): v for k, v in sd["model_state_dict"].items()}

            # remove the last layer     
            for key in list(pretrained_state_dict.keys()):
                if 'head' in key:
                    del pretrained_state_dict[key]

            model.load_state_dict(pretrained_state_dict, strict=False)
        
        sigmoid = torch.nn.Sigmoid()

        if self.hparams.freezeBackbone:

            logging.info("Freeze backbone")
            for param in model.parameters():
                param.requires_grad = False

     

        return model, embedding, sigmoid


    def forward(self, input1, input2):

        output1 = self.model(input1)
        output2 = self.model(input2)


        output_con = torch.cat((output1, output2), 1)

        #print(output_con.shape)

        output_before_sig = self.embedding(output_con)

        output = self.sigmoid(output_before_sig)

        return output

    def training_step(self, batch, batch_idx):
        x0, x1, y, similarity  = batch


        output = self(x0, x1)

        loss_criterion = torch.nn.BCELoss(weight =similarity)
        loss = loss_criterion(output, y)

        self.log("train_loss", loss, prog_bar=True, logger=True, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):

        #logging.info("start validation step")

        x0, x1, target, similarity = batch
        output_model = self(x0, x1)

        #print(x0.size())
        #print(similarity.size())
        #print(target.size())

        loss_criterion = torch.nn.BCELoss(weight =similarity)
        loss = loss_criterion(output_model, target)
    
        #loss = self.criterion(output_model, target)

        correct = 0

        pred = torch.where(output_model > 0.5, 1, 0)
        correct += pred.eq(target.view_as(pred)).sum().item()

        val_acc = 100. * correct / len(output_model)

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

        DatasetFolder_Train = torchvision.datasets.ImageFolder(
            self.hparams.imageFolderTrain)
        #logging.info(f"Build train")
        #  imageFolderDataset, database_csv_File, transform=None, num_pairs=25600)    
        dataset = SiameseNetworkDataset(imageFolderDataset=DatasetFolder_Train, transform=tfm_train,database_csv_File=self.hparams.database_csv,database_vipp_file=self.hparams.database_vipp,similarity_training=self.hparams.similarity_training, num_pairs=self.hparams.num_pairs)

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

        DatasetFolder_Valid = torchvision.datasets.ImageFolder(self.hparams.imageFolderValid)
        #logging.info(f"Build validation")
        dataset = SiameseNetworkDataset(imageFolderDataset=DatasetFolder_Valid, transform=tfm_valid,database_csv_File=self.hparams.database_csv,database_vipp_file=self.hparams.database_vipp,similarity_training=self.hparams.similarity_training, num_pairs= int (self.hparams.num_pairs ) )
        #logging.info(f"Finish validation")
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
        )

        #self.total_number_training_images = len(dataloader.dataset)
        #logging.info(f"\nThe total number of samples : {self.total_number_training_images}")
        #logging.info('#####################################################################')
        #logging.info('#####################################################################')
        return dataloader


def parse_args():
    args = ArgumentParser()
    args.add_argument("-c", "--config", type=Path,
                      default='config/siamese_resnet101_sigmoid.yml')
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
