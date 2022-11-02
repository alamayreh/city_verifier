import torch
import logging
import torchvision
import random
from argparse import  ArgumentParser
from pathlib import Path
from utils import *
from PIL import Image
from train_contrastive import SiameseNetwork as SiameseNetwork_contrastive
from train_sigmoid import SiameseNetwork as SiameseNetwork_sigmoid
from train_sigmoid import SiameseNetwork, SiameseNetworkDataset
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import accuracy_score
from tqdm import tqdm

# export CUDA_VISIBLE_DEVICES=4,5,6,7
# python3 inference_per_city.py --city_class Moscow --gpu 
def parse_args():
    args = ArgumentParser()
    args.add_argument(
        "--checkpoint_sigmoid",
        type=Path,
        #default=Path("/data/omran/cities_data/models/resnet101_128_sigmoid_Nonlinearty_pretrain_VIPPGeo/221011-1211/ckpts/epoch_94.ckpt"),
        #default=Path("models/resnet101_128_sigmoid_acc_noFlatren/220915-0730/ckpts/epoch_15.ckpt"),
        default=Path("/data/omran/cities_data/models/resnet101_64_sigmoid_Nonlinearty_freezeBackbone/221029-0428/ckpts/epoch_568.ckpt"),
        help="Checkpoint to already trained model (*.ckpt)",
    )
    args.add_argument(
        "--hparams_sigmoid",
        type=Path,
        #default=Path("/data/omran/cities_data/models/resnet101_128_sigmoid_acc_pretrain_ImgNet/220916-0358/tb_logs/version_0/hparams.yaml"),
        default=Path("/data/omran/cities_data/models/resnet101_64_sigmoid_Nonlinearty_freezeBackbone/221029-0428/tb_logs/version_0/hparams.yaml"),
        help="Path to hparams file (*.yaml) generated during training",
    )
    args.add_argument(
        "--checkpoint_contrastive",
        type=Path,
        #default=Path("models/resnet101_128_embedding/220912-0404/ckpts/epoch_15.ckpt"),
        #default=Path("/data/omran/cities_data/models/resnet101_64_embedding/220912-0923/ckpts/epoch_34.ckpt"),
        #default=Path("models/resnet101_64_embedding_Adam/220919-0156/ckpts/epoch_100.ckpt"),
        #default=Path("models/resnet101_32_embedding_SGD/220919-0838/ckpts/epoch_1858.ckpt"),
        #default=Path('models/resnet101_1024_embedding_margin_1_SGD/220920-1046/ckpts/epoch_346.ckpt'),
        default=Path('/data/omran/cities_data/models/resnet101_64_embedding_25600_freezeBackbone/221028-1010/ckpts/epoch_183.ckpt'),
        help="Checkpoint to already trained model (*.ckpt)",
    )
    args.add_argument(
        "--hparams_contrastive",
        type=Path,
        #default=Path("models/resnet101_128_embedding/220912-0404/tb_logs/version_0/hparams.yaml"),
        #default=Path("/data/omran/cities_data/models/resnet101_64_embedding/220912-0923/tb_logs/version_0/hparams.yaml"),
        #default=Path("models/resnet101_32_embedding_SGD/220919-0838/tb_logs/version_0/hparams.yaml"),
        default=Path('/data/omran/cities_data/models/resnet101_64_embedding_25600_freezeBackbone/221028-1010/tb_logs/version_0/hparams.yaml'),
        help="Path to hparams file (*.yaml) generated during training",
    )
    args.add_argument(
        "--image_dir",
        type=Path,
        default=Path("/data/omran/cities_data/dataset/cities/test"),
        #default=Path("/data/omran/cities_data/dataset/open_set"),
        help="Folder containing test set images.",
    )
    args.add_argument(
        "--city_class",
        type=str,
        help="Folder containing test set images per class.",
    )
    
    # environment
    args.add_argument(
        "--gpu",
        action="store_true",
        help="Use GPU for inference if CUDA is available",
    )
    args.add_argument("--batch_size", type=int, default=64)
    args.add_argument(
        "--num_workers",
        type=int,
        default=16,
        help="Number of workers for image loading and pre-processing",
    )
    return args.parse_args()



class SiameseNetworkDataset_one_class(Dataset):

    def __init__(self, imageFolderDataset, transform=None,city=None,num_pairs=4000):
        self.imageFolderDataset = imageFolderDataset
        self.transform = transform
        self.city = city
        self.num_pairs = num_pairs
        self.same_pair = 0


    def __getitem__(self, index):

        dict = self.imageFolderDataset.class_to_idx 
        city_class = dict[self.city]

        should_get_same_class = random.randint(0, 1)

        if should_get_same_class and (self.same_pair < self.num_pairs/2) :
            while True:
                img1_tuple = random.choice(self.imageFolderDataset.imgs)
                img0_tuple = random.choice(self.imageFolderDataset.imgs)
                if (img0_tuple[1] == img1_tuple[1]) and (city_class == img0_tuple[1]) :
                    self.num_pairs +=1                    
                    break
        else:
            while True:
                
                # keep looping till a different class image is found

                img1_tuple = random.choice(self.imageFolderDataset.imgs)
                img0_tuple = random.choice(self.imageFolderDataset.imgs)

                if (img0_tuple[1] != img1_tuple[1]) and (city_class == img0_tuple[1]) :
                    #print(f"{img0_tuple[1]},{img1_tuple[1]},{city_class}")
                    break

        img0 = Image.open(img0_tuple[0])
        img1 = Image.open(img1_tuple[0])


        if self.transform is not None:
            img0 = self.transform(img0)
            img1 = self.transform(img1)
        

        return img0, img1, torch.from_numpy(np.array([int(img1_tuple[1] != img0_tuple[1])], dtype=np.float32))

        

    def __len__(self):
        return self.num_pairs
        #return len(self.imageFolderDataset.imgs)




def test_dataloader(image_dir, batch_size, num_workers,city_class):

    # logging.info("val_dataloader")

    tfm_test = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize(224),
            torchvision.transforms.CenterCrop(224),
            torchvision.transforms.ToTensor(),
        ]
    )

    DatasetFolder_test = torchvision.datasets.ImageFolder(image_dir)

    dataset = SiameseNetworkDataset_one_class(imageFolderDataset=DatasetFolder_test, transform=tfm_test,city =city_class )
    #dataset = SiameseNetworkDataset(imageFolderDataset=DatasetFolder_test, transform=tfm_test )

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



    logging.info(f"Loading test data : {args.image_dir}")

    test_dataloader = test_dataloader(args.image_dir, args.batch_size, args.num_workers,args.city_class)

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

        for im1,im2,target in tqdm(test_dataloader):
            if args.gpu:
                im1 = im1.cuda()
                im2 = im2.cuda()
                target = target.cuda()    

            y_true_temp = target.cpu().detach().numpy()
            for i in y_true_temp:
                y_true.append(i[:])
            #output_model = model(Variable(x0).cuda(), Variable(x1).cuda())
            output_model = model_sigmoid(im1, im2)
        
            #print(output_model)
        
            pred = torch.where(output_model > 0.5, 1, 0)
            

            y_pred_temp = pred.cpu().detach().numpy()
            for i in y_pred_temp:
                y_pred.append(i[:])

            correct += pred.eq(target.view_as(pred)).sum().item()


        #print('y_true', y_true)
        #print('y_pred', y_true)

        print(confusion_matrix(y_true, y_pred)) 

        print(f'acc skealrn just to check {accuracy_score(y_true, y_pred) * 100 }')
        val_acc_sigmoid = 100. * correct / dataset_length

        print(f"val_acc sigmoid per {args.city_class}: {val_acc_sigmoid}")

    
    if (sigmoid != True):

        logging.info(f"Loading model from : {args.checkpoint_contrastive}")
        logging.info(f"Accurcy per city :   {args.city_class}")

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

        for im1,im2,target in tqdm(test_dataloader):
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


        val_acc_contrastive =100. * correct / dataset_length

        val_acc_cosine = 100. * correct_cosine / dataset_length

        print(f"val_acc contrastive {args.city_class}: {val_acc_contrastive}")

        print(f"val_acc cosine {args.city_class}: {val_acc_cosine}")