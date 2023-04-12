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
        default=Path("/data/omran/cities_data/dataset/filtered/dataset_10k/test_100_restricted"),
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

class SiameseNetworkDataset(Dataset):

    def __init__(self, database_csv, database_folder, image_path, image_prob, IMG_ID_test, transform=None):


        self.database_folder = database_folder
        self.image_path      = image_path
        self.image_prob      = image_prob
        self.IMG_ID_test     = IMG_ID_test
        self.transform       = transform
        self.database_csv    = database_csv
        self.images_database = listdir(database_folder)

 
    def __len__(self):

        return len(self.images_database)
    
    def string_to_prob(string_prob):

        # Read probability from datafram 
        image_prob_str = ((string_prob)[1:])[:-1].split()
        image_prob = [float(i) for i in image_prob_str]
        
        return image_prob

    def distance_euclidean(self,prob_0,prob_1):

        eDistance = math.dist((prob_0),(prob_1))

        return (eDistance)

    def distance_cos(self,prob_0,prob_1):

        cDistance = spatial.distance.cosine(prob_0, prob_1)

        return (1-cDistance)
   
    def __getitem__(self, index):
     
        IMG_ID_base = self.images_database[index]
        #print(f"image_database {IMG_ID_base}")
        
        img0_prob_str = (self.database_csv.loc[IMG_ID_base].S16)
        img0_prob     = string_to_prob(img0_prob_str)

 
        distance = self.distance_cos(img0_prob, self.image_prob)
        similarity      = torch.from_numpy(np.array([distance],dtype=np.float32))

        #print(f'euclidean_distance: {euclidean_distance} and similarity : {similarity}')

        
        img0 = Image.open(str(self.database_folder) + '/' + IMG_ID_base)
        img1 = Image.open(self.image_path)


        if self.transform is not None:
            img0 = self.transform(img0)
            img1 = self.transform(img1)

       
        return self.IMG_ID_test, IMG_ID_base, img0, img1, similarity




def test_dataloader(database_csv,image_dir_database,image_path,image_prob,IMG_ID_test):

    tfm_test = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize(224),
            torchvision.transforms.CenterCrop(224),
            torchvision.transforms.ToTensor(),
        ]
    )

    #DatasetFolder_test = torchvision.datasets.ImageFolder(image_dir_database)

    dataset = SiameseNetworkDataset(database_csv=database_csv,database_folder=image_dir_database,image_path=image_path,image_prob=image_prob, IMG_ID_test=IMG_ID_test,transform=tfm_test)    

    return dataset

def string_to_prob(string_prob):

    # Read probability from datafram
    image_prob_str = ((string_prob)[1:])[:-1].split()
    image_prob = [float(i) for i in image_prob_str]

    return image_prob

if __name__ == '__main__':

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


    in_city = ["Amsterdam", "Barcelona", "Berlin", "London", "NewYork", "LosAngeles", "Rome", "Milan", "Paris", "Tokyo"]
    out_city = ["Amman", "Istanbul", "Mexico_city", "Singapore", "Quebec", "Vancouver", "Venice", "Florence"]
    
    y_pred = []
    y_true = []
    for test_city in in_city+out_city:
        # Read cities from dir test 
        
        sim_whole_closeset = []
        for database_city in in_city:
           
            logging.info(f"Test {test_city} city on {database_city} database")
        
            db_similarity =  pd.read_csv(args.S16_csv, usecols=['IMG_ID','S16']).set_index('IMG_ID')
        
            # ream metadata files 
            if test_city in in_city:
                test_dir = join(args.image_dir_test, test_city)
            else:
                test_dir = join('/data/omran/cities_data/dataset/filtered/open_set_restricted', test_city)
                
            
            image_dir    = join(args.image_dir_database, database_city)
        
            # construct dataloader 
            test_dataset_list = []
        
            logging.info("Building dataloader")
            
            #for index, row in tqdm(test_image_db.iterrows(),  total=(test_image_db.shape[0])):
            for image_path in tqdm(listdir(test_dir)): # this loop loads 101 closed set images
                image_path = str(join(test_dir, image_path)) 
        
                IMG_ID_test =  image_path.split('/')[-1]
                #print(image_path)
        
                image_prob_str =  db_similarity.loc[IMG_ID_test].S16
                image_prob     =  string_to_prob (image_prob_str)            
        
        
                test_dataset_one_image = test_dataloader(db_similarity,image_dir,image_path,image_prob,IMG_ID_test)
                test_dataset_list.append(test_dataset_one_image)
        
        
            test_dataset = torch.utils.data.ConcatDataset(test_dataset_list)
        
            dataloader = torch.utils.data.DataLoader(
                test_dataset,
                batch_size=args.batch_size,
                num_workers= args.num_workers,
                pin_memory=True,
            )
        
            dataset_length = len(dataloader.dataset)
            logging.info(f"Number of images pairs: {dataset_length}")
        
            if dataset_length == 0:
                raise RuntimeError(f"No images found in {args.image_dir_database} city {args.database_city}")
                        
            #first = True   
            
            p_same_sum = 0 
            p_diff_sum = 0 
        
            out_db =  pd.DataFrame(columns=['IMG_ID_test','IMG_ID_data_base','similarity','sigmoid_output','probablity_same','probablity_diff'])
        
            sim_whole_set = []
            ids = 1
            idx = 0
            ref_num = 9
            for IMG_ID_test, IMG_ID_data_base, im1, im2, w_similarity in tqdm(dataloader):
                # this loop run similarity between test set and each images in database set
                # Question: what is database set?
                #print(IMG_ID_test, IMG_ID_data_base)
                #if ids == 5:
                #    break
                #ids += 1
                idx += 1
                
                if 101 * (ids - 1) + ref_num < idx < ids * 101:
                    continue
                
                if idx == ids * 101:
                    ids += 1
                
                if args.gpu:
                    im1 = im1.cuda()
                    im2 = im2.cuda()
                    w_similarity = w_similarity.cuda()
        
        
                output_model = model_sigmoid(im1, im2)
        
                # JUN you NEED this line 
                
                output_model = output_model.detach()
                
                # in training 0 -> same class, and 1 diff class 
                # if the probablity < 0.5 vote 1 
                    # same                                     # diff
                p_same = (torch.ones_like(output_model)-output_model) * w_similarity 
                p_diff = (output_model) * w_similarity
                
                p_score = output_model.cpu().detach().numpy()
                #p_score [p_score > 0.5] = 1
                #p_score [p_score <= 0.5] = 0
                sim_whole_set += p_score.tolist()
                #print(np.array(sim_whole_set).shape)
        
                IMG_ID_test      = np.array(IMG_ID_test).reshape(-1,1)
                IMG_ID_data_base = np.array(IMG_ID_data_base).reshape(-1,1)
        
               
                one_patch = np.hstack((IMG_ID_test,IMG_ID_data_base,w_similarity.cpu().detach().numpy(),output_model.cpu().detach().numpy(),p_same.cpu().detach().numpy(),p_diff.cpu().detach().numpy()))
                
        
                
                out_db = pd.concat([out_db, pd.DataFrame(one_patch,columns=['IMG_ID_test','IMG_ID_data_base','similarity','sigmoid_output','probablity_same','probablity_diff'])], axis=0)
                #print(out_db)
        
        
            #print(np.array(sim_whole_set).shape)  # should be (N_test_images x N_datasase_images) x 1
            sim_whole_set = np.array(sim_whole_set).reshape(ref_num+1, 101).transpose()
            #print(np.array(sim_whole_set).shape)
            out_db.reset_index()
        
            if not os.path.exists(args.output_dir):
                os.makedirs(args.output_dir)    
        
            out_db.to_csv(f'{args.output_dir}/{test_city}_on_{args.database_city}_database_openset.csv',index=False)
            
            sim_whole_closeset.append(sim_whole_set.mean(axis=1).tolist()) # should be N_citybase x N_test_images x 1
            #print(np.array(sim_whole_closeset).shape)
        
        
        if test_city in in_city:
            y_true += np.zeros(len(sim_whole_set)).tolist()
        else:
            y_true += np.ones(len(sim_whole_set)).tolist()
            
        y_pred += (np.array(sim_whole_closeset).sum(axis=0)).tolist()  # should be (N_cities x N_test_images) x 1
        
        #print(np.array(y_true).shape)
        #print(np.array(y_pred).shape)
    AUC = roc_auc_score(y_true, y_pred)
    print(f"Open Set AUROC : {AUC}")
        