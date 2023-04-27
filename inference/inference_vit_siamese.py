import sys
sys.path.insert(0, '/data/omran/siamese_cities')
import torch
import random
import logging
import torchvision
from PIL import Image
from argparse import ArgumentParser
from pathlib import Path
from utils import *
#from training.train_sigmoid import SiameseNetwork as SiameseNetwork_sigmoid
#from training.train_contrastive import SiameseNetwork as SiameseNetwork_contrastive
from tqdm import tqdm
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'DeJavu Serif'
plt.rcParams['font.serif'] = ['Times New Roman']

#### import the dataset class used in training #### 

#from training.train_sigmoid_vipp_balance import SiameseNetworkDataset as SiameseNetworkDataset
from training.train_vit_siamese import SiameseNetworkDataset as SiameseNetworkDataset
from training.train_vit_siamese import SiameseNetwork as SiameseNetwork_sigmoid

# export CUDA_VISIBLE_DEVICES=4,5,6,7
# python3 inference_vit_siamese.py --gpu --GeoVIPP

def parse_args():
    args = ArgumentParser()
    args.add_argument(
        "--checkpoint_sigmoid",
        type=Path,

        #default=Path("/data/omran/cities_data/models/dataset_10k/resnet50/VippTraing_CityPretrainImageNe_NoFreezeBackbone/230203-1158/epoch_25.ckpt"),     

        #default=Path("/data/omran/cities_data/models/dataset_10k/vit/pretrain_andrea_lr_0.01/230327-1036/ckpts/epoch_4.ckpt"),
        
        #default=Path("/data/omran/cities_data/models/dataset_10k/vit/pretrain_imagenet_32_No_GeoVIPP/230412-0838/epoch_37.ckpt"),
        #default=Path("/data/omran/cities_data/models/dataset_10k/vit/pretrain_imagenet_56_batch/230328-0931/ckpts/epoch_34.ckpt"),   
        
        #GeoVipp_50
        default=Path("/data/omran/cities_data/models/dataset_10k/vit/last_exp/pretrain_ImgNet_GeoVIPP_50/230420-0451/ckpts/epoch_49.ckpt"),          

 
        help="Checkpoint to already trained model (*.ckpt)",
    )
    args.add_argument(
        "--hparams_sigmoid",
        type=Path,

        #default=Path("/data/omran/cities_data/models/dataset_10k/resnet50/VippTraing_CityPretrainImageNe_NoFreezeBackbone/230203-1158/tb_logs/version_0/hparams.yaml"),

        #default=Path("/data/omran/cities_data/models/dataset_10k/vit/pretrain_andrea_lr_0.01/230327-1036/tb_logs/version_0/hparams.yaml"),

        #default=Path("/data/omran/cities_data/models/dataset_10k/vit/pretrain_imagenet_32_No_GeoVIPP/230412-0838/tb_logs/version_0/hparams.yaml"),

        #default=Path("/data/omran/cities_data/models/dataset_10k/vit/pretrain_imagenet_56_batch/230328-0931/tb_logs/version_0/hparams.yaml"),

        default=Path("/data/omran/cities_data/models/dataset_10k/vit/last_exp/pretrain_ImgNet_GeoVIPP_50/230420-0451/tb_logs/version_0/hparams.yaml"),


        help="Path to hparams file (*.yaml) generated during training",
    )
    args.add_argument(
        "--image_dir",
        type=Path,
        #default=Path("/data/omran/cities_data/dataset/cities/test"),
        #default=Path("/data/omran/cities_data/dataset/open_set"),
        #default=Path("/data/omran/cities_data/dataset/filtered/test"),
        default=Path("/data/omran/cities_data/dataset/filtered/dataset_10k/test"),
        help="Folder containing test set images.",
    )
    
    args.add_argument(
        "--S16_csv",
        type=Path,
        default=Path("/data/omran/cities_data/dataset/S16_database_10k.csv"), 
        #default=Path("/data/omran/cities_data/dataset/S16_database_open_set.csv"), 
        help="CSV folder for images database.",
    )
    # environment
    
    args.add_argument(
        "--vipp_database",
        type=Path,
        default=Path("/data/omran/cities_data/dataset/Vipp_classes_10k.csv"),
        #default=Path("/data/omran/cities_data/dataset/Vipp_classes_open_set.csv"),
        help="Folder containing CSV files meta data for all images.",
    )
    args.add_argument(
        "--gpu",
        action="store_true",
        help="Use GPU for inference if CUDA is available",
    )
    args.add_argument("--batch_size", type=int, default=6) #32 #768
    args.add_argument(
        "--num_workers",
        type=int,
        default=2,
        help="Number of workers for image loading and pre-processing",
    )

    args.add_argument(
        "--thr",
        type=float,
        default=0.5,
        help="Test",
    )
    args.add_argument(
        "--GeoVIPP",
        action="store_true",
        help="Use GeoVIPP in sampling",
    )    
    return args.parse_args()


class SiameseNetworkDataset_basic(Dataset):

    def __init__(self, imageFolderDataset, transform=None, num_pairs=51200):
        self.imageFolderDataset = imageFolderDataset
        self.transform = transform
        self.num_pairs = num_pairs

        self.dict = self.imageFolderDataset.class_to_idx
        self.dict_class = {v: k for k, v in self.dict.items()}
        logging.info(f"Class dictionary : \n {self.dict_class}")

    def __getitem__(self, index):

        #img0_tuple = random.choice(self.imageFolderDataset.imgs)

        #should_get_same_class = random.randint(0, 1)

        if (index < self.num_pairs/2):
            while True:
                img0_tuple = random.choice(self.imageFolderDataset.imgs)
                img1_tuple = random.choice(self.imageFolderDataset.imgs)
                
                if img0_tuple[1] == img1_tuple[1]:
                    # print("nowBreak")
                    break
        else:
            while True:
                # keep looping till a different class image is found
                img0_tuple = random.choice(self.imageFolderDataset.imgs)
                img1_tuple = random.choice(self.imageFolderDataset.imgs)
               
                if img0_tuple[1] != img1_tuple[1]:
                    break

        img0 = Image.open(img0_tuple[0])
        img1 = Image.open(img1_tuple[0])

        if self.transform is not None:
            img0 = self.transform(img0)
            img1 = self.transform(img1)

        return img0, img1, torch.from_numpy(np.array([int(img1_tuple[1] != img0_tuple[1])], dtype=np.float32)), self.dict_class[img0_tuple[1]], self.dict_class[img1_tuple[1]]

    def __len__(self):
        # return len(self.imageFolderDataset.imgs)
        return self.num_pairs


def test_dataloader(image_dir, batch_size, num_workers):

    # logging.info("val_dataloader")

    tfm_test = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize(224),
            torchvision.transforms.CenterCrop(224),
            torchvision.transforms.ToTensor(),
        ]
    )

    DatasetFolder_test = torchvision.datasets.ImageFolder(image_dir)


    dataset = SiameseNetworkDataset(imageFolderDataset=DatasetFolder_test, transform=tfm_test,database_csv_File=args.S16_csv,database_vipp_file=args.vipp_database, similarity_training=False,geovipp_training = args.GeoVIPP, num_pairs=64000)


    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
    )

    return dataloader


if __name__ == '__main__':

    args = parse_args()

    logging.basicConfig(level=logging.INFO)

    logging.info(f"Loading test data  : {args.image_dir}")

    test_dataloader = test_dataloader(
        args.image_dir, args.batch_size, args.num_workers)

    dataset_length = len(test_dataloader.dataset)
    logging.info(f"Number of images: {dataset_length}")

    if len(test_dataloader.dataset) == 0:
        raise RuntimeError(f"No images found in {args.image_dir}")

    sigmoid = True

    if (sigmoid == True):

        logging.info(f"Loading model from : {args.checkpoint_sigmoid}")

        model_sigmoid = SiameseNetwork_sigmoid.load_from_checkpoint(
            checkpoint_path=str(args.checkpoint_sigmoid),
            hparams_file=str(args.hparams_sigmoid),
            map_location=None,
        )

        model_sigmoid.eval()

        if args.gpu and torch.cuda.is_available():
            model_sigmoid.cuda()

        correct = 0
        y_true = []
        y_pred = []

        y_score = []

        #for im1, im2, target, city_1, city_2 in tqdm(test_dataloader):
        
        for im1, im2, target, _ in tqdm(test_dataloader):    
            if args.gpu:
                im1 = im1.cuda()
                im2 = im2.cuda()
                target = target.cuda()

            ## For confusion matrix 
            y_true_temp = target.cpu().detach().numpy()
            for i in y_true_temp:
                y_true.append(i[:])    

            output_model = model_sigmoid(im1, im2)

            # For AUC
            y_score_temp = output_model.cpu().detach().numpy()

            for j in y_score_temp:
                y_score.append(j[:])

            pred = torch.where(output_model > args.thr, 1, 0)

            # For confusion matrix 
            y_pred_temp = pred.cpu().detach().numpy()
            for i in y_pred_temp:
                y_pred.append(i[:])


            correct = correct + pred.eq(target.view_as(pred)).sum().item()


        print(confusion_matrix(y_true, y_pred))
        print(confusion_matrix(y_true, y_pred, normalize='true'))
        #print(confusion_matrix(y_true, y_pred, normalize='all'))

        print(f'acc skealrn just to check {accuracy_score(y_true, y_pred) * 100 }')
        val_acc_sigmoid = 100. * correct / dataset_length
        print(f"val_acc sigmoid: {val_acc_sigmoid}")

        print('----------------------------------------------')
        
        #print(f'y_score : {y_score}')
        #print(f'y_true : {y_true}')
        print(f'AUC score {roc_auc_score(y_true, y_score)}')
       
        fpr, tpr, threshold = roc_curve(y_true, y_score)
        #print(fpr)
        #print(tpr)
        #print(threshold)
        
        plt.plot(fpr, tpr,color="teal")

        plt.xlabel('False Positive Rate', fontsize=14,labelpad=5)
        plt.ylabel('True Positive Rate', fontsize=14,labelpad=5)
        plt.savefig('roc_curve.pdf',dpi=300)  
        plt.savefig('roc_curve.svg')  