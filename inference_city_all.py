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

#export CUDA_VISIBLE_DEVICES=4,5,6,7

def parse_args():
    args = ArgumentParser()
    args.add_argument(
        "--checkpoint_sigmoid",
        type=Path,
        default=Path("/data/omran/cities_data/models/resnet101_128_sigmoid_Nonlinearty_pretrain_VIPPGeo/221011-1211/ckpts/epoch_94.ckpt"),
        #default=Path("models/resnet101_128_sigmoid_acc_noFlatren/220915-0730/ckpts/epoch_15.ckpt"),
        help="Checkpoint to already trained model (*.ckpt)",
    )
    args.add_argument(
        "--hparams_sigmoid",
        type=Path,
        default=Path("/data/omran/cities_data/models/resnet101_128_sigmoid_acc_pretrain_ImgNet/220916-0358/tb_logs/version_0/hparams.yaml"),
        help="Path to hparams file (*.yaml) generated during training",
    )
    args.add_argument(
        "--checkpoint_contrastive",
        type=Path,
        #default=Path("models/resnet101_128_embedding/220912-0404/ckpts/epoch_15.ckpt"),
        default=Path("/data/omran/cities_data/models/resnet101_64_embedding/220912-0923/ckpts/epoch_34.ckpt"),
        #default=Path("models/resnet101_64_embedding_Adam/220919-0156/ckpts/epoch_100.ckpt"),
        #default=Path("models/resnet101_32_embedding_SGD/220919-0838/ckpts/epoch_1858.ckpt"),
        #default=Path('models/resnet101_1024_embedding_margin_1_SGD/220920-1046/ckpts/epoch_346.ckpt'),
        help="Checkpoint to already trained model (*.ckpt)",
    )
    args.add_argument(
        "--hparams_contrastive",
        type=Path,
        #default=Path("models/resnet101_128_embedding/220912-0404/tb_logs/version_0/hparams.yaml"),
        default=Path("/data/omran/cities_data/models/resnet101_64_embedding/220912-0923/tb_logs/version_0/hparams.yaml"),
        #default=Path("models/resnet101_32_embedding_SGD/220919-0838/tb_logs/version_0/hparams.yaml"),
        help="Path to hparams file (*.yaml) generated during training",
    )
    args.add_argument(
        "--image_dir",
        type=Path,
        default=Path("/data/omran/cities_data/dataset/cities/test"),
        #default=Path("/data/omran/cities_data/dataset/open_set"),
        help="Folder containing test set images.",
    )
    # environment
    args.add_argument(
        "--gpu",
        action="store_true",
        help="Use GPU for inference if CUDA is available",
    )
    args.add_argument("--batch_size", type=int, default=1)
    args.add_argument(
        "--city_class",
        type=str,
        help="Folder containing test set images per class.",
    )
    
    args.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of workers for image loading and pre-processing",
    )
    return args.parse_args()

class SiameseNetworkDataset(Dataset):

    def __init__(self, imageFolderDataset, transform=None):
        self.imageFolderDataset = imageFolderDataset
        self.transform = transform

        self.dict = self.imageFolderDataset.class_to_idx 

        self.dict_class = {v: k for k, v in self.dict.items()}
        logging.info(f"Class dictionary : \n {self.dict_class}")
       

    def __getitem__(self, index):


        img0_tuple = random.choice(self.imageFolderDataset.imgs)

        should_get_same_class = random.randint(0, 1)

        if should_get_same_class:
            while True:
                img1_tuple = random.choice(self.imageFolderDataset.imgs)
                if img0_tuple[1] == img1_tuple[1]:
                    # print("nowBreak")
                    break
        else:
            while True:
                # keep looping till a different class image is found

                img1_tuple = random.choice(self.imageFolderDataset.imgs)
                if img0_tuple[1] != img1_tuple[1]:
                    break

        img0 = Image.open(img0_tuple[0])
        img1 = Image.open(img1_tuple[0])


        if self.transform is not None:
            img0 = self.transform(img0)
            img1 = self.transform(img1)

        return img0, img1, torch.from_numpy(np.array([int(img1_tuple[1] != img0_tuple[1])], dtype=np.float32)), self.dict_class[img0_tuple[1]],self.dict_class[img1_tuple[1]]

    def __len__(self):
        return len(self.imageFolderDataset.imgs)


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

    dataset = SiameseNetworkDataset(
        imageFolderDataset=DatasetFolder_test, transform=tfm_test)

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

    logging.info(f"Loading model from : {args.checkpoint_sigmoid}")

    model_sigmoid = SiameseNetwork_sigmoid.load_from_checkpoint(
        checkpoint_path=str(args.checkpoint_sigmoid),
        hparams_file=str(args.hparams_sigmoid),
        map_location=None,
    )

    model_sigmoid.eval()

    if args.gpu and torch.cuda.is_available():
        model_sigmoid.cuda()

    
    logging.info(f"Loading test data : {args.image_dir}")

    test_dataloader = test_dataloader(args.image_dir, args.batch_size, args.num_workers)

    dataset_length = len(test_dataloader.dataset)
    logging.info(f"Number of images: {dataset_length}")

    if len(test_dataloader.dataset) == 0:
        raise RuntimeError(f"No images found in {args.image_dir}")

    correct = 0 

    num_samples = 0
    
    num_same_class = 0 
    num_diff_class  = 0

    for im1,im2,target,city_1,city_2  in tqdm(test_dataloader):

        if (city_1[0] !=  args.city_class):

            
            continue
        
      

        #print((target[0][0]))
        if args.gpu:
            im1 = im1.cuda()
            im2 = im2.cuda()
            target = target.cuda()    

        #output_model = model(Variable(x0).cuda(), Variable(x1).cuda())
        output_model = model_sigmoid(im1, im2)
        
        #print(output_model)
        
        pred = torch.where(output_model > 0.5, 1, 0)

        #print('pred', pred)
        #print('target', target)
        correct += pred.eq(target.view_as(pred)).sum().item()

        num_samples+=1 

    val_acc_sigmoid =100. * correct / num_samples



    logging.info(f"Loading model from : {args.checkpoint_contrastive}")

    model_contrastive = SiameseNetwork_contrastive.load_from_checkpoint(
        checkpoint_path=str(args.checkpoint_contrastive),
        hparams_file=str(args.hparams_contrastive),
        map_location=None,
    )

    model_contrastive.eval()
    
    if args.gpu and torch.cuda.is_available():
        model_contrastive.cuda()
    
    correct = 0 
    correct_cosine = 0

    cos = torch.nn.CosineSimilarity()

    num_samples = 0
    num_same_class = 0 
    num_diff_class  = 0
    
    for im1,im2,target,city_1,city_2 in tqdm(test_dataloader):

        if (city_1[0] != args.city_class):
          
            continue
        
        #print(f"city 1 : {city_1} and city 2 : {city_2}")
        num_samples +=1 

        if(target == 0 ):
            num_same_class+=1
        if(target ==1 ):
            num_diff_class+=1

        if args.gpu:
            im1 = im1.cuda()
            im2 = im2.cuda()
            target = target.cuda()    

        output1, output2 = model_contrastive(im1, im2)
       
        euclidean_distance = torch.nn.functional.pairwise_distance(output1, output2)

        cosine_distance =  torch.tensor([1]).cuda() - cos(output1, output2)

        pred_cosine =  torch.where(cosine_distance > 0.5, 1, 0)

        pred = torch.where(euclidean_distance > 0.5, 1, 0)

        correct += pred.eq(target.view_as(pred)).sum().item()

        correct_cosine += pred_cosine.eq(target.view_as(pred_cosine)).sum().item()


    val_acc_contrastive =100. * correct / num_samples

    val_acc_cosine = 100. * correct_cosine / num_samples



    print('#################################################')
    print(f'CITY : {args.city_class}')
    print(f'Number of samples                : {num_samples}') 
    print(f'Number of samples per same city  : {num_same_class}')
    print(f'Number of samples per diff city  : {num_diff_class}')

    print(f"val_acc sigmoid                  : {val_acc_sigmoid}")
    print(f"val_acc contrastive              : {val_acc_contrastive}")
    print(f"val_acc cosine                   : {val_acc_cosine}")
    print('#################################################')