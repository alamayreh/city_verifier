## FOR JUN ###
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
import csv

from sklearn.metrics import roc_curve, auc, roc_auc_score
from pytorch_ood.utils import OODMetrics, ToUnknown
from pytorch_ood.detector import OpenMax
from glob import glob
import os

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)


def dataloader(imageFolderTest,batch_size,num_workers):

        tfm_test = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize(256),
                torchvision.transforms.CenterCrop(224),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(
                    (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
                ),
            ]
        )

        DatasetFolder_Test = torchvision.datasets.ImageFolder(root=imageFolderTest,transform=tfm_test)
        #print(DatasetFolder_Test.class_to_idx)
        
        dataloader = torch.utils.data.DataLoader(
            DatasetFolder_Test,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True,
            shuffle=False,
        )

        return dataloader

def datafolder(imageFolderTest):

        tfm_test = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize(256),
                torchvision.transforms.CenterCrop(224),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(
                    (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
                ),
            ]
        )

        DatasetFolder = torchvision.datasets.ImageFolder(root=imageFolderTest,transform=tfm_test)

        return DatasetFolder

def datafolder_out(imageFolderTest):

        tfm_test = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize(256),
                torchvision.transforms.CenterCrop(224),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(
                    (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
                ),
            ]
        )

        DatasetFolder = torchvision.datasets.ImageFolder(root=imageFolderTest,transform=tfm_test,target_transform=ToUnknown())

        return DatasetFolder
        

def parse_args():
    args = ArgumentParser()
    args.add_argument("-c", "--config", type=Path)
    args.add_argument("--progbar", action="store_true")
    args.add_argument(
        "--checkpoint",
        type=Path,
        #default=Path('/data/omran/cities_data/models/city_classifier/dataset10k/resnet50/230203-0739/ckpts/epoch_38.ckpt'),
        default=Path("/data/omran/cities_data/models/city_classifier/dataset10k/resnet101_GeoVIPP/230211-1245/epocl_1.ckpt"),           
        help="Checkpoint to already trained model (*.ckpt)",
    )
    
    args.add_argument(
        "--hparams",
        type=Path,
        #default=Path("/data/omran/cities_data/models/city_classifier/dataset10k/resnet50/230203-0739/tb_logs/version_0/hparams.yaml"),
        default=Path("/data/omran/cities_data/models/city_classifier/dataset10k/resnet101_GeoVIPP/230211-1245/tb_logs/version_0/hparams.yaml"),           
        help="Checkpoint to already trained model (*.ckpt)",
    )
    
    args.add_argument(
        "--test_dir",
        type=Path,
        default=Path("/data/omran/cities_data/dataset/filtered/dataset_10k/validation"),
        help="This is the test dir closed set",
    ) 
    
    args.add_argument(
        "--test_out_dir",
        type=Path,
        default=Path("/data/omran/cities_data/dataset/filtered/open_set_restricted"),   
        help="This is the test dir for open set",
    ) 

    args.add_argument(
        "--training",
        type=Path,
        default=Path("/data/omran/cities_data/dataset/filtered/dataset_10k/training"),   
        help="Checkpoint to already trained model (*.ckpt)",
    ) 
    return args.parse_args()


if __name__ == '__main__':
    tfm_test = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize(256),
                torchvision.transforms.CenterCrop(224),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(
                    (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
                ),
            ]
        )
        
    args = parse_args()

    logging.basicConfig(level=logging.INFO)

    logging.info(f"Loading model from : {args.checkpoint}")

    model = Classifier.load_from_checkpoint(
        checkpoint_path=str(args.checkpoint),
        hparams_file=str(args.hparams),
        map_location=None,
    )



    if torch.cuda.is_available():
        model.cuda()

    model.eval()
    # Read cities from dir test 

    logging.info("Building dataloader")

    logging.info(f"Loading test data   : {args.test_dir}")

    test_dataloader = dataloader(args.test_dir,batch_size=24,num_workers=1)
    
    logging.info(f"Loading open test data   : {args.test_out_dir}")

    test_open_dataloader = dataloader(args.test_out_dir,batch_size=24,num_workers=1)

    dataset_length = len(test_dataloader.dataset)
    logging.info(f"Number of images: {dataset_length}")

    if len(test_dataloader.dataset) == 0:
        raise RuntimeError(f"No images found in {args.image_dir}")

    correct = 0
    in_city = ['Amsterdam', 'Barcelona', 'Berlin', 'London', 'LosAngeles', 'Milan', 'NewYork', 'Paris', 'Rome', 'Tokyo']
    y_score = []
    y_true = []
    target_cvs = f'closedset_valid10k_scores.csv'
    with open(target_cvs, 'w') as cvsFile:
        writer = csv.writer(cvsFile)
        writer.writerow(['img_path', 'Amsterdam', 'Barcelona', 'Berlin', 'London', 'LosAngeles', 'Milan', 'NewYork', 'Paris', 'Rome', 'Tokyo', 'GT', 'Pred_correct'])
        for label, city in enumerate(in_city):
            target = torch.Tensor([label]).unsqueeze(0)
            closed_set = glob(os.path.join(args.test_dir, city, '*.*'))
            #for im, target in tqdm(test_dataloader):    
            for img_path in closed_set:
                im = Image.open(img_path)
                im = tfm_test(im).unsqueeze(0)
                im = im.cuda()
                target = target.cuda()
        
                output_model = model(im)
                
                probs = torch.softmax(output_model, dim=1)
                logits, _ = output_model.max(dim=1)
                
                y_score += logits.detach().cpu().numpy().tolist()
                y_true += np.ones(im.shape[0]).tolist()
                
                pred = probs.argmax(dim=1)
                correct += pred.eq(target.view_as(pred)).sum().item()
                if pred.eq(target.view_as(pred)).sum().item() == 1:
                    pred_idx = 1
                else:
                    pred_idx = 0
                idx = output_model.detach().cpu().numpy().tolist()
                #print(len(idx[0]))
                writer.writerow([img_path] + idx[0] + [label] + [pred_idx])
                #writer.writerow([img_path, idx[0], idx[1], idx[2], idx[3], idx[4], idx[5], idx[6], idx[7], idx[8], idx[9], label, pred_idx])

        acc = 100. * correct / dataset_length
        print(f"acc : {acc}")