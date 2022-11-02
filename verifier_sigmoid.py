import torch
import random
import logging
import torchvision
from PIL import Image
from argparse import  ArgumentParser
from pathlib import Path
from utils import *
from train_sigmoid import SiameseNetwork as SiameseNetwork_sigmoid
from train_contrastive import SiameseNetwork as SiameseNetwork_contrastive
from tqdm import tqdm
from os import listdir
from os.path import isfile, join

#python3 verifier_sigmoid.py --gpu
#export CUDA_VISIBLE_DEVICES=3

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
        "--image_dir",
        type=Path,
        default=Path("/data/omran/cities_data/dataset/cities/training/Tokyo"),
        #default=Path("/data/omran/cities_data/dataset/open_set"),
        #default=Path("/data/omran/cities_data/dataset/cities/test_10_images_ready/Cairo"),
        help="Folder containing test set images of the database.",
    )
    args.add_argument(
        "--image_dir_test",
        type=Path,
        #default=Path("/data/omran/cities_data/dataset/cities/training/Cairo"),
        #default=Path("/data/omran/cities_data/dataset/open_set"),
        default=Path("/data/omran/cities_data/dataset/cities/test"),
        help="Folder containing test set images of the database.",
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
        default=2,
        help="Number of workers for image loading and pre-processing",
    )
    return args.parse_args()

class SiameseNetworkDataset(Dataset):

    def __init__(self, imageFolder,image_path, transform=None):

        self.imageFolder = imageFolder
        self.transform   = transform
        self.image_path  = image_path


        # Images of the databse to compare with 
        self.images_Folder_files = [f for f in listdir(self.imageFolder) if isfile(join(self.imageFolder, f))]

        #logging.info(f"Image Folder :  {self.imageFolder} and Image Path {self.image_path}")
       

    def __getitem__(self, index):

        #print(f'index : {index}')

        #one_image = (str(self.imageFolder) + '/' + self.images_Folder_files[index])
        
        #print(f'image path           : {one_image}')
        #print(f'image to verfiy path : {self.image_path}')

        #print('#################################################')

        img0 = Image.open(str(self.imageFolder) + '/' + self.images_Folder_files[index])
        img1 = Image.open(self.image_path)


        if self.transform is not None:
            img0 = self.transform(img0)
            img1 = self.transform(img1)

        return img0, img1

    def __len__(self):

        return len(listdir(self.imageFolder))


def test_dataloader(images_dir,image_path, batch_size, num_workers):

    # logging.info("val_dataloader")

    tfm_test = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize(224),
            torchvision.transforms.CenterCrop(224),
            torchvision.transforms.ToTensor(),
        ]
    )

    #DatasetFolder_test = torchvision.datasets.ImageFolder(image_dir)



    dataset = SiameseNetworkDataset(imageFolder=images_dir,image_path=image_path, transform=tfm_test)    

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
    logging.info(f"Loading base data  : {args.image_dir}")

    model_sigmoid = SiameseNetwork_sigmoid.load_from_checkpoint(
        checkpoint_path=str(args.checkpoint_sigmoid),
        hparams_file=str(args.hparams_sigmoid),
        map_location=None,
    )

    model_sigmoid.eval()

    if args.gpu and torch.cuda.is_available():
        model_sigmoid.cuda()


    for filename in listdir(args.image_dir_test):

        logging.info(f"Test {filename} on {args.image_dir} database")
        test_dir = join(args.image_dir_test, filename)

        num_images_verified = 0

        for image in tqdm(listdir(test_dir)):
            
            image_path = join(test_dir, image)

            #logging.info(f"test image : {image_path}")

            test_dataloader_one_image = test_dataloader(args.image_dir,image_path, args.batch_size, args.num_workers)

            dataset_length = len(test_dataloader_one_image.dataset)
            #logging.info(f"Number of images: {dataset_length}")

            if len(test_dataloader_one_image.dataset) == 0:
                raise RuntimeError(f"No images found in {args.image_dir}")

            correct = 0 

            

            for im1,im2  in (test_dataloader_one_image):

                if args.gpu:
                    im1 = im1.cuda()
                    im2 = im2.cuda()


                output_model = model_sigmoid(im1, im2)
        

                # in training 0 -> same class, and 1 diff class 
                # if the probablity < 0.5 vote 1 
        
                pred = torch.where(output_model < 0.5, 1, 0)

                correct +=  (pred).sum().item()


            

            votes = 100. * correct / dataset_length

            if(votes > 50):
                num_images_verified += 1


            #print('#################################################')
            #print(f'Number of votes                  : {correct}')
            #print(f"votes  percent                   : {votes} %")
       
        print(f'Number of images verified {filename}             : {num_images_verified}')
        print('#################################################')