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

from pytorch_ood.utils import OODMetrics, ToUnknown
from pytorch_ood.detector import OpenMax


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

        dataloader = torch.utils.data.DataLoader(
            DatasetFolder_Test,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True,
            shuffle=True,
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
        default=Path("/data/omran/cities_data/dataset/filtered/dataset_10k/test_100_restricted"),   
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

    test_dataloader = dataloader(args.test_dir,batch_size=32,num_workers=1)

    dataset_length = len(test_dataloader.dataset)
    logging.info(f"Number of images: {dataset_length}")

    if len(test_dataloader.dataset) == 0:
        raise RuntimeError(f"No images found in {args.image_dir}")

    correct = 0

    y_true = []
    y_pred = []

    for im, target in tqdm(test_dataloader):    

        im = im.cuda()
        target = target.cuda()

        output_model = model(im)

        probs = torch.softmax(output_model, dim=1)
        pred = probs.argmax(dim=1)
        correct += pred.eq(target.view_as(pred)).sum().item()

    acc = 100. * correct / dataset_length
    print(f"acc : {acc}")

    print('--------------------------------------------------------------------')

    logging.info(f"Loading training data : {args.training}")

    training_dataloader = dataloader(args.training,batch_size=32,num_workers=1)

    logging.info(f"Loading test data close and open dataset : {args.test_dir} and {args.test_out_dir}")


    #test_dataloader_closed_open = dataloader('/data/omran/cities_data/dataset/filtered/test_closed_open_set',batch_size=32,num_workers=1) 
    dataset_in_test  = datafolder(args.test_dir)
    dataset_out_test = datafolder_out(args.test_out_dir)

    

    test_dataloader_closed_open = torch.utils.data.DataLoader(dataset_in_test + dataset_out_test,
            batch_size=32,
            num_workers=1,
            pin_memory=True,
            shuffle=True,)


    device = "cuda:0"


    detector = OpenMax(model, tailsize=25, alpha=5, euclid_weight=0.5)
    detector.fit(training_dataloader, device=device)

    metrics = OODMetrics()

    threshold = 0.5 

    correct_in_out = 0
    len_in_out = len(test_dataloader_closed_open.dataset) 

    for x, y in test_dataloader_closed_open:

        output = (detector(x.to(device)))

        #print('y',y)
        #print('output',output)
        metrics.update(output, y)


        out_th = torch.threshold(output, threshold, 0)

        #print("out_th",out_th)

        pred_in_out = torch.where(out_th == 0, torch.tensor(0), torch.tensor(1))

        #print("pred_in_out",pred_in_out)
        
        target_in_out = torch.where(y == -1, torch.tensor(1), torch.tensor(0))

        #print("target_in_out",target_in_out)

        correct_in_out += pred_in_out.eq(target_in_out.view_as(pred_in_out)).sum().item()


    print(metrics.compute())

    print(f'acc openset detector = {correct_in_out / len_in_out}')