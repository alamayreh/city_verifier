## This is for Jun ##

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
from training.train_vit_siamese import SiameseNetwork as SiameseNetwork_sigmoid
from tqdm import tqdm
from os import listdir
from os.path import isfile, join
import os
import math
from scipy import spatial
from sklearn.metrics import roc_curve, auc, roc_auc_score
from glob import glob
import csv
from PIL import Image
from scipy.special import softmax

#[Moscow,London,Shanghai,Cairo,Delhi,New_york,Rio_de_Janeiro,Sydney,Roma,Tokyo]
#export CUDA_VISIBLE_DEVICES=4
#python3 verifier_vit_openset.py --test_city Tokyo --database_city Tokyo
#export CUDA_VISIBLE_DEVICES=4

def parse_args():
    args = ArgumentParser()
    args.add_argument(
        "--checkpoint_sigmoid",
        type=Path,
        #default=Path("/data/omran/cities_data/models/dataset_10k/vit/pretrain_andrea_lr_0.01/230327-1036/ckpts/epoch_4.ckpt"),

        default=Path("/data/omran/cities_data/models/dataset_10k/vit/pretrain_imagenet_56_batch/230328-0931/ckpts/epoch_34.ckpt"),      

        help="Checkpoint to already trained model (*.ckpt)",
    )
    args.add_argument(
        "--hparams_sigmoid",
        type=Path,

        #default=Path("/data/omran/cities_data/models/dataset_10k/vit/pretrain_andrea_lr_0.01/230327-1036/tb_logs/version_0/hparams.yaml"),

        default=Path("/data/omran/cities_data/models/dataset_10k/vit/pretrain_imagenet_56_batch/230328-0931/tb_logs/version_0/hparams.yaml"),

        help="Path to hparams file (*.yaml) generated during training",
    )

    args.add_argument(
        "--output_dir",
        type=Path,
        default=Path("/data/omran/cities_data/results/dataset_10k/ViT_Siamese_openset_10"),
        help="Folder contains the output.",
    )  

    args.add_argument(
        "--image_dir_database",
        type=Path,
        #default=Path("/data/omran/cities_data/dataset/filtered/dataset_10k/test_100_restricted"),
        default=Path("/data/omran/cities_data/dataset/filtered/dataset_10k/validation"),
        #default=Path("/data/omran/cities_data/dataset/filtered/open_set_restricted"),
        help="Folder contians database images.",
    )    

    args.add_argument(
        "--image_dir_test",
        type=Path,
        default=Path("/data/omran/cities_data/dataset/filtered/dataset_10k/test_100_restricted"),
        #default=Path("/data/omran/cities_data/dataset/filtered/open_set_restricted"),
        help="Folder containing CSV files meta data for of test images.",
    )

    args.add_argument(
        "--S16_csv",
        type=Path,
        default=Path("/data/omran/cities_data/dataset/S16_database_10k.csv"), 
        help="CSV folder for images database.",
    )  

    args.add_argument(
        "--database_city",
        type=str,
        help="Database city",
    )



    args.add_argument(
        "--test_city",
        type=str,
        help="Test",
    )

    # environment
    args.add_argument(
        "--gpu",
        action="store_true",
        default='--gpu',
        help="Use GPU for inference if CUDA is available",
    )
    args.add_argument("--batch_size", type=int, default=1) #40 #64
   
    args.add_argument(
        "--num_workers",
        type=int,
        default=8,
        help="Number of workers for image loading and pre-processing",
    )
    return args.parse_args()


if __name__ == '__main__':
    
    tfm_test = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize(224),
            torchvision.transforms.CenterCrop(224),
            torchvision.transforms.ToTensor(),
        ]
    )
    
    ### start with reading prediction results of validation 10k set
    target_cvs = 'closedset_valid10k_scores.csv'
    
    csvFile = open(target_cvs, "r")
    reader = csv.reader(csvFile)
    
    label = 0
    count = 0
    valid_10k = []
    for row in reader:
        if reader.line_num == 1:
            continue
            
        count += 1
        rows = [float(item) for item in row[1:]]
        rows.append(row[0])
        valid_10k.append(rows)
        
    
    args = parse_args()

    logging.basicConfig(level=logging.INFO)

    logging.info(f"Loading model from : {args.checkpoint_sigmoid}")
    logging.info(f"Loading base data  : {args.image_dir_database} city {args.database_city}")

    model_sigmoid = SiameseNetwork_sigmoid.load_from_checkpoint(
        checkpoint_path=str(args.checkpoint_sigmoid),
        hparams_file=str(args.hparams_sigmoid),
        map_location=None,
    )

    model_sigmoid.eval()

    if args.gpu and torch.cuda.is_available():
        model_sigmoid.cuda()


    in_city = ['Amsterdam', 'Barcelona', 'Berlin', 'London', 'LosAngeles', 'Milan', 'NewYork', 'Paris', 'Rome', 'Tokyo']
    out_city = ["Amman", "Istanbul", "Mexico_city", "Singapore", "Quebec", "Vancouver", "Venice", "Florence"]
    
    y_pred = []
    y_true = []
    num_refs = 10
    th_refs = 0.99
    for test_city in in_city + out_city:
        # Read cities from dir test 
        
        sim_whole_closeset = []
        for idx_label, database_city in enumerate(in_city):
           
            logging.info(f"Test {test_city} city on {database_city} database")
            
            database_img_list = glob(os.path.join(args.image_dir_database, database_city, '*.*'))
            #print(len(database_img_list))
            print(len(valid_10k),idx_label*1000)
            valid_ref = valid_10k[int(idx_label*1000):int((idx_label+1)*1000)]
            #print(len(valid_ref))
            ### now I need to select the reference images for verification rejection
            ref_img_path = []
            ref_count = 0
            for item_idx, item in enumerate(valid_ref):
                #[float(item) for item in row]
                logit = item[0:10]
                prob = softmax(logit)
                #print(logit, prob)
                #print(prob[np.argmax(prob)])
                prediction = item[11]
                #print(prediction)
                if ref_count == num_refs:
                    continue
                    
                if prediction == 1 and prob[np.argmax(prob)] > th_refs:
                    #print(database_img_list[item_idx])
                    #print(item[12])
                    ref_img_path.append(item[12])
                    ref_count += 1
            #print(ref_img_path)
            
            # ream metadata files 
            if test_city in in_city:
                test_dir = join(args.image_dir_test, test_city)
            else:
                test_dir = join('/data/omran/cities_data/dataset/filtered/open_set_restricted', test_city)
        
            # construct dataloader 
            test_dataset_list = []
        
            logging.info("Building dataloader")
            
            sim_whole_set = []
            #for index, row in tqdm(test_image_db.iterrows(),  total=(test_image_db.shape[0])):
            for image_path in tqdm(listdir(test_dir)): # this loop loads 101 closed set images
                image_path = str(join(test_dir, image_path)) 
        
                IMG_ID_test =  image_path.split('/')[-1]
                #print(image_path)
                im1 = Image.open(image_path)
                im1 = tfm_test(im1).unsqueeze(0)
                sim_refs = []
                for ref_img in ref_img_path:
                    #print(ref_img, image_path)
                    im2 = Image.open(ref_img)
                    im2 = tfm_test(im2).unsqueeze(0)
                    
                    if args.gpu:
                        im1 = im1.cuda()
                        im2 = im2.cuda()
                    
                    output_model = model_sigmoid(im1, im2)
                    #print(output_model)
                    p_score = output_model.cpu().detach().numpy()
                    #p_score [p_score > 0.5] = 1
                    #p_score [p_score <= 0.5] = 0
                    sim_refs += p_score.tolist()
                    
                sim_whole_set.append(sim_refs)
                    
            #print(np.array(sim_whole_set).shape)  # should be (N_test_images x N_datasase_images) x 1
            sim_whole_set = np.array(sim_whole_set).reshape(101, num_refs)
            #print((sim_whole_set).shape)
            
            sim_whole_closeset.append(sim_whole_set.mean(axis=1).tolist()) # should be N_citybase x N_test_images x 1
        
        
        #print(np.array(sim_whole_closeset).shape)
        if test_city in in_city:
            y_true += np.zeros(len(sim_whole_set)).tolist()
        else:
            y_true += np.ones(len(sim_whole_set)).tolist()
            
        y_pred += (np.max(np.array(sim_whole_closeset), axis=0)).tolist()
        # should be (N_cities x N_test_images) x 1
        
    print(np.array(y_true).shape)
    print(np.array(y_pred).shape)
    AUC = roc_auc_score(y_true, y_pred)
    print(f"Open Set AUROC : {AUC}")
        