import torch
import random
import logging
import torchvision
import pandas as pd
from PIL import Image
from argparse import  ArgumentParser
from pathlib import Path
from utils import *
from train_sigmoid import SiameseNetwork as SiameseNetwork_sigmoid
from train_contrastive import SiameseNetwork as SiameseNetwork_contrastive
from tqdm import tqdm
from os import listdir
from os.path import isfile, join

#python3 verifier_sigmoid_365.py --gpu
#export CUDA_VISIBLE_DEVICES=2

def parse_args():
    args = ArgumentParser()
    args.add_argument(
        "--checkpoint_sigmoid",
        type=Path,
        default=Path("/data/omran/cities_data/models/resnet101_64_sigmoid_Nonlinearty_freezeBackbone/221029-0428/ckpts/epoch_568.ckpt"),
        help="Checkpoint to already trained model (*.ckpt)",
    )
    args.add_argument(
        "--hparams_sigmoid",
        type=Path,
        default=Path("/data/omran/cities_data/models/resnet101_64_sigmoid_Nonlinearty_freezeBackbone/221029-0428/tb_logs/version_0/hparams.yaml"),
        help="Path to hparams file (*.yaml) generated during training",
    )

    args.add_argument(
        "--database_csv",
        type=Path,
        default=Path("/data/omran/cities_data/dataset/cities/csv_meta/training/Tokyo.csv"), #Rio_de_Janeiro
        help="CSV file for images database.",
    )

    args.add_argument(
        "--image_dir_database",
        type=Path,
        default=Path("/data/omran/cities_data/dataset/cities/training/Tokyo"),
        help="Folder contians database images.",
    )

    args.add_argument(
        "--image_csv_test",
        type=Path,
        default=Path("/data/omran/cities_data/dataset/cities/csv_meta/test"),
        help="Folder containing CSV files meta data for of test images.",
    )

    args.add_argument(
        "--image_dir_test",
        type=Path,
        default=Path("/data/omran/cities_data/dataset/cities/test"),
        help="Folder containing CSV files meta data for of test images.",
    )
    # environment
    args.add_argument(
        "--gpu",
        action="store_true",
        help="Use GPU for inference if CUDA is available",
    )
    args.add_argument("--batch_size", type=int, default=128)

    args.add_argument(
        "--city_class",
        type=str,
        help="Folder containing test set images per class.",
    )
    
    args.add_argument(
        "--num_workers",
        type=int,
        default=1,
        help="Number of workers for image loading and pre-processing",
    )
    return args.parse_args()

class SiameseNetworkDataset(Dataset):

    def __init__(self, database_csv, database_folder, image_path, image_prob, transform=None):


        self.database_folder = database_folder
        self.image_path      = image_path
        self.image_prob      = image_prob
        self.transform       = transform

        self.cos             = torch.nn.CosineSimilarity()
        self.database_csv    = pd.read_csv(database_csv)#.reset_index(drop=True)

 
    def __len__(self):

        return self.database_csv.shape[0]

    def __getitem__(self, index):
     
        #print('#################################################')
        #print(f'index : {index}')
        #print(self.database_csv.iloc[index])        
        #print(self.database_csv.iloc[index].IMG_ID)        
        #print(img0_prob)   
        
        img0_prob_str = ((self.database_csv.iloc[index].Probabily_365)[1:])[:-1].split(' ')
        img0_prob     = [float(i) for i in img0_prob_str]

        img0_probality = torch.FloatTensor([img0_prob])
        img1_probality = torch.FloatTensor([self.image_prob])

        euclidean_distance = torch.nn.functional.pairwise_distance(img0_probality, img1_probality)
        #cos_distance    = self.cos(img0_probality, img1_probality)       
        #similarity      = torch.ones_like(cos_distance) - cos_distance      

        similarity      = 1 / euclidean_distance 

        #print(f'euclidean_distance: {euclidean_distance} and similarity : {similarity}')

        
        img0 = Image.open(str(self.database_folder) + '/' + self.database_csv.iloc[index].IMG_ID)
        img1 = Image.open(self.image_path)


        if self.transform is not None:
            img0 = self.transform(img0)
            img1 = self.transform(img1)

        return img0, img1, similarity




def test_dataloader(database_csv,image_dir_database,image_path,image_prob, batch_size, num_workers):

    tfm_test = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize(224),
            torchvision.transforms.CenterCrop(224),
            torchvision.transforms.ToTensor(),
        ]
    )

    #DatasetFolder_test = torchvision.datasets.ImageFolder(image_dir_database)

    dataset = SiameseNetworkDataset(database_csv=database_csv,database_folder=image_dir_database,image_path=image_path,image_prob=image_prob, transform=tfm_test)    

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
    )

    return dataloader

def get_percent_vote_per_image():

    return 0 

if __name__ == '__main__':

    args = parse_args()

    logging.basicConfig(level=logging.INFO)

    logging.info(f"Loading model from : {args.checkpoint_sigmoid}")
    logging.info(f"Loading base data  : {args.image_dir_database}")

    model_sigmoid = SiameseNetwork_sigmoid.load_from_checkpoint(
        checkpoint_path=str(args.checkpoint_sigmoid),
        hparams_file=str(args.hparams_sigmoid),
        map_location=None,
    )

    model_sigmoid.eval()

    if args.gpu and torch.cuda.is_available():
        model_sigmoid.cuda()


    # Read cities from dir test 
    for filename in listdir(args.image_dir_test):
    #for filename in ['Moscow']:

        logging.info(f"Test {filename} on {args.image_dir_database} database")

        test_dir = join(args.image_dir_test, filename)

        num_images_verified = 0

        test_csv_file = str(join(args.image_csv_test, filename)) + '.csv'

        test_image_db = pd.read_csv(test_csv_file)


        for index, row in tqdm(test_image_db.iterrows(),  total=(test_image_db.shape[0])):
            
            image_path = str(join(test_dir, row.IMG_ID)) 

            image_prob_str =((row.Probabily_365)[1:])[:-1].split(' ')
            image_prob = [float(i) for i in image_prob_str]

            #logging.info(f"test image : {image_path}")
            #logging.info(f"test image  image_prob: {type(image_prob[0])}")

            test_dataloader_one_image = test_dataloader(args.database_csv,args.image_dir_database,image_path,image_prob, args.batch_size, args.num_workers)

            dataset_length = len(test_dataloader_one_image.dataset)
            #logging.info(f"Number of images: {dataset_length}")

            if len(test_dataloader_one_image.dataset) == 0:
                raise RuntimeError(f"No images found in {args.image_dir_database}")
                

            correct    = 0           
            p_same_sum = 0 
            p_diff_sum = 0 

            for im1,im2, w_similarity  in (test_dataloader_one_image):
                     
                if args.gpu:
                    im1 = im1.cuda()
                    im2 = im2.cuda()
                    w_similarity = w_similarity.cuda()


                output_model = model_sigmoid(im1, im2)

                output_model = output_model.detach()
                # in training 0 -> same class, and 1 diff class 
                # if the probablity < 0.5 vote 1 
                    # same                                     # diff
                p_same = (torch.ones_like(output_model)-output_model) * w_similarity 
                p_diff = (output_model) * w_similarity

                #print(f'output_model : {output_model}')
                #print(f'w_similarity : {w_similarity}')
                #print(f'p_same : {p_same}')
                #print(f'p_diff : {p_diff}')
                

                p_same_sum += p_same.sum().item()
                p_diff_sum += p_diff.sum().item()

                #pred = torch.where(output_model < 0.5, 1, 0)

                #correct +=  (pred).sum().item()
         

            #votes = 100. * correct / dataset_length

            if(p_same_sum > p_diff_sum):
                num_images_verified += 1
                #print(f'The image belong to the databset')

       
        print(f'Number of images verified {filename}             : {num_images_verified}')
        print('#################################################')