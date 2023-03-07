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
from training.train_sigmoid_vipp import SiameseNetwork as SiameseNetwork_sigmoid
from tqdm import tqdm
from os import listdir
from os.path import isfile, join
import math
from scipy import spatial

#[Moscow,London,Shanghai,Cairo,Delhi,New_york,Rio_de_Janeiro,Sydney,Roma,Tokyo]

#python3 verifier_sigmoid_filtered.py --test_city Tokyo --database_city Tokyo
#export CUDA_VISIBLE_DEVICES=4

def parse_args():
    args = ArgumentParser()
    args.add_argument(
        "--checkpoint_sigmoid",
        type=Path,
        #default=Path("/data/omran/cities_data/models/resnet101_64_sigmoid_VIPP_Freeze_Filtered_No_Similarity/221130-1344/ckpts/epock_119.ckpt"),
        #default=Path("/data/omran/cities_data/models/Filtered_15/NoVippTraing_CityPretrain_NoFreezeBackbone/221216-0930/ckpts/epoch_24.ckpt"),
        #default=Path("/data/omran/cities_data/models/Filtered_15/VippTraing_CityPretrainImageNe_NoFreezeBackbone/221222-0949/ckpts/epoch_6.ckpt"),
        #default=Path("/data/omran/cities_data/models/Filtered_15/VippTraing_CityPretrainImageNe_NoFreezeBackbone_balanced/230108-0853/ckpts/epoch_8.ckpt"),
        #default=Path("/data/omran/cities_data/models/Filtered_15/VippTraing_VippPretrain_NoFreezeBackbone_balanced_50/230109-0849/ckpts/epock_16.ckpt"),        
        #default=Path("/data/omran/cities_data/models/Filtered_15/VippTraing_CityPretrainImageNe_NoFreezeBackbone_NY_LOS_25/230111-1220/ckpts/epoch_28.ckpt"),        
        
        #default=Path("/data/omran/cities_data/models/Filtered_15/VippTraing_CityPretrainImageNe_NoFreezeBackbone/221223-0921/ckpts/epoch_89.ckpt"),        
        default=Path("/data/omran/cities_data/models/dataset_10k/resnet50/VippTraing_CityPretrainImageNe_NoFreezeBackbone/230203-1158/epoch_25.ckpt"),        
        #default=Path("/data/omran/cities_data/models/dataset_10k/resnet101/VippTraing_CityPretrainVIPP_NoFreezeBackbone/epoch_399.ckpt"),        

        help="Checkpoint to already trained model (*.ckpt)",
    )
    args.add_argument(
        "--hparams_sigmoid",
        type=Path,
        #default=Path("/data/omran/cities_data/models/resnet101_64_sigmoid_VIPP_Freeze_Filtered_No_Similarity/221130-1344//tb_logs/version_0/hparams.yaml"),
        #default=Path("/data/omran/cities_data/models/Filtered_15/NoVippTraing_CityPretrain_NoFreezeBackbone/221216-0930/tb_logs/version_0/hparams.yaml"),
        #default=Path("/data/omran/cities_data/models/Filtered_15/VippTraing_CityPretrainImageNe_NoFreezeBackbone/221222-0949/tb_logs/version_0/hparams.yaml"),
        #default=Path("/data/omran/cities_data/models/Filtered_15/VippTraing_CityPretrainImageNe_NoFreezeBackbone_balanced/230108-0853/tb_logs/version_0/hparams.yaml"),
        #default=Path("/data/omran/cities_data/models/Filtered_15/VippTraing_VippPretrain_NoFreezeBackbone_balanced_50/230109-0849/tb_logs/version_0/hparams.yaml"),
        #default=Path("/data/omran/cities_data/models/Filtered_15/VippTraing_CityPretrainImageNe_NoFreezeBackbone_NY_LOS_25/230111-1220/tb_logs/version_0/hparams.yaml"),

        #default=Path("/data/omran/cities_data/models/Filtered_15/VippTraing_CityPretrainImageNe_NoFreezeBackbone/221223-0921//tb_logs/version_0/hparams.yaml"),
        default=Path("/data/omran/cities_data/models/dataset_10k/resnet50/VippTraing_CityPretrainImageNe_NoFreezeBackbone/230203-1158/tb_logs/version_0/hparams.yaml"),



        help="Path to hparams file (*.yaml) generated during training",
    )

    args.add_argument(
        "--S16_csv",
        type=Path,
        default=Path("/data/omran/cities_data/dataset/S16_database_10k.csv"), 
        help="CSV folder for images database.",
    )
    
    args.add_argument(
        "--image_dir_database",
        type=Path,
        default=Path("/data/omran/cities_data/dataset/filtered/open_set_restricted"),
        help="Folder contians database images.",
    )

    args.add_argument(
        "--database_city",
        type=str,
        help="Database city",
    )

    args.add_argument(
        "--image_dir_test",
        type=Path,
        default=Path("/data/omran/cities_data/dataset/filtered/open_set_restricted"),
        help="Folder containing CSV files meta data for of test images.",
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
    args.add_argument("--batch_size", type=int, default=65) #40
   
    args.add_argument(
        "--num_workers",
        type=int,
        default=1,
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


    # Read cities from dir test 
   
    logging.info(f"Test {args.test_city} city on {args.database_city} database")

    db_similarity =  pd.read_csv(args.S16_csv, usecols=['IMG_ID','S16']).set_index('IMG_ID')

    # ream metadata files 
    test_dir      = join(args.image_dir_test, args.test_city)
    
    image_dir    = join(args.image_dir_database, args.database_city)

    # construct dataloader 
    test_dataset_list = []

    logging.info("Building dataloader")

    #for index, row in tqdm(test_image_db.iterrows(),  total=(test_image_db.shape[0])):
    for image_path in tqdm(listdir(test_dir)):
            
        image_path = str(join(test_dir, image_path)) 

        IMG_ID_test =  image_path.split('/')[-1]

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

    for IMG_ID_test, IMG_ID_data_base, im1,im2, w_similarity  in tqdm(dataloader):
                     
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


        IMG_ID_test      = np.array(IMG_ID_test).reshape(-1,1)
        IMG_ID_data_base = np.array(IMG_ID_data_base).reshape(-1,1)

       
        one_patch = np.hstack((IMG_ID_test,IMG_ID_data_base,w_similarity.cpu().detach().numpy(),output_model.cpu().detach().numpy(),p_same.cpu().detach().numpy(),p_diff.cpu().detach().numpy()))
        

        
        out_db = pd.concat([out_db, pd.DataFrame(one_patch,columns=['IMG_ID_test','IMG_ID_data_base','similarity','sigmoid_output','probablity_same','probablity_diff'])], axis=0)


    out_db.reset_index()
    out_db.to_csv(f'/data/omran/cities_data/results/dataset_10k/ResNet50_ImageNetT_VippTraining_test_100_restricted/{args.test_city}_on_{args.database_city}_database.csv',index=False)
