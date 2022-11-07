from email.policy import default
import torch
import random
import logging
import torchvision
from PIL import Image
from argparse import ArgumentParser
from pathlib import Path
from utils import *
from train_sigmoid import SiameseNetwork as SiameseNetwork_sigmoid
from train_contrastive import SiameseNetwork as SiameseNetwork_contrastive
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

# export CUDA_VISIBLE_DEVICES=4,5,6,7


def parse_args():
    args = ArgumentParser()
    args.add_argument(
        "--checkpoint_sigmoid",
        type=Path,
        # default=Path("/data/omran/cities_data/models/resnet101_128_sigmoid_Nonlinearty_pretrain_VIPPGeo/221011-1211/ckpts/epoch_94.ckpt"),
        # default=Path("/data/omran/cities_data/models/resnet101_128_sigmoid_acc_noFlatren/220915-0730/ckpts/epoch_15.ckpt"),
        #default=Path("/data/omran/cities_data/models/resnet101_128_sigmoid_acc_pretrain_ImgNet/220916-0358/ckpts/epoch_621.ckpt"),
        default=Path("/data/omran/cities_data/models/resnet101_64_sigmoid_Nonlinearty_freezeBackbone/221029-0428/ckpts/epoch_568.ckpt"),
        help="Checkpoint to already trained model (*.ckpt)",
    )
    args.add_argument(
        "--hparams_sigmoid",
        type=Path,
        # default=Path("/data/omran/cities_data/models/resnet101_128_sigmoid_acc_pretrain_ImgNet/220916-0358/tb_logs/version_0/hparams.yaml"),
        # default=Path("/data/omran/cities_data/models/resnet101_128_sigmoid_acc_noFlatren/220915-0730/tb_logs/version_0/hparams.yaml"),
        #default=Path("/data/omran/cities_data/models/resnet101_128_sigmoid_acc_pretrain_ImgNet/220916-0358/tb_logs/version_0/hparams.yaml"),
        default=Path("/data/omran/cities_data/models/resnet101_64_sigmoid_Nonlinearty_freezeBackbone/221029-0428/tb_logs/version_0/hparams.yaml"),
        help="Path to hparams file (*.yaml) generated during training",
    )
    
    args.add_argument(
        "--checkpoint_contrastive",
        type=Path,
        #default=Path("/data/omran/cities_data/models/resnet101_128_embedding/220912-0404/ckpts/epoch_15.ckpt"),
        #default=Path("/data/omran/cities_data/models/resnet101_64_embedding/220912-0923/ckpts/epoch_34.ckpt"),
        #default=Path("/data/omran/cities_data/models/resnet101_64_embedding_Adam/220919-0156/ckpts/epoch_100.ckpt"),
        #default=Path("/data/omran/cities_data/models/resnet101_32_embedding_SGD/220919-0838/ckpts/epoch_1858.ckpt"),
        #default=Path('/data/omran/cities_data/models/resnet101_1024_embedding_margin_1_SGD/220920-1046/ckpts/epoch_346.ckpt'),
        #default=Path('/data/omran/cities_data/models/resnet101_16_embedding_margin_1_SGD/220919-0909/ckpts/epoch_285.ckpt'),
        #default=Path('/data/omran/cities_data/models/resnet101_64_embedding_no_pretrain_SGD_256000_samples/221026-1235/ckpts/epoch_03.ckpt'),
        default=Path('/data/omran/cities_data/models/resnet101_64_embedding_25600_samples/221028-0506/ckpts/epoch_13.ckpt'),
        #default=Path('/data/omran/cities_data/models/resnet101_64_embedding_25600_freezeBackbone/221028-1010/ckpts/epoch_183.ckpt'),
        help="Checkpoint to already trained model (*.ckpt)",
    )
    args.add_argument(
        "--hparams_contrastive",
        type=Path,
        #default=Path("/data/omran/cities_data/models/resnet101_64_embedding_Adam/220919-0156/tb_logs/version_0/hparams.yaml"),
        #default=Path("/data/omran/cities_data/models/resnet101_32_embedding_SGD/220919-0838/tb_logs/version_0/hparams.yaml"),
        #default=Path("/data/omran/cities_data/models/resnet101_128_embedding/220912-0404/tb_logs/version_0/hparams.yaml"),
        #default=Path("/data/omran/cities_data/models/resnet101_64_embedding/220912-0923/tb_logs/version_0/hparams.yaml"),
        #default=Path("/data/omran/cities_data/models/resnet101_1024_embedding_margin_1_SGD/220920-1046/tb_logs/version_0/hparams.yaml"),
        #default=Path('/data/omran/cities_data/models/resnet101_16_embedding_margin_1_SGD/220919-0909/tb_logs/version_0/hparams.yaml'),
        #default=Path('/data/omran/cities_data/models/resnet101_64_embedding_no_pretrain_SGD_256000_samples/221026-1235/tb_logs/version_0/hparams.yaml'),
        default=Path('/data/omran/cities_data/models/resnet101_64_embedding_25600_samples/221028-0506/tb_logs/version_0/hparams.yaml'),
        #default=Path('/data/omran/cities_data/models/resnet101_64_embedding_25600_freezeBackbone/221028-1010/tb_logs/version_0/hparams.yaml'),
        help="Path to hparams file (*.yaml) generated during training",
    )
    args.add_argument(
        "--image_dir",
        type=Path,
        default=Path("/data/omran/cities_data/dataset/cities/test"),
        #default=Path("/data/omran/cities_data/dataset/open_set"),
        #default=Path("/data/omran/cities_data/dataset/cities/validation"),
        help="Folder containing test set images.",
    )
    # environment
    args.add_argument(
        "--gpu",
        action="store_true",
        help="Use GPU for inference if CUDA is available",
    )
    args.add_argument("--batch_size", type=int, default=512)
    args.add_argument(
        "--num_workers",
        type=int,
        default=6,
        help="Number of workers for image loading and pre-processing",
    )
    return args.parse_args()


class SiameseNetworkDataset(Dataset):

    def __init__(self, imageFolderDataset, transform=None, num_pairs=None):
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

        for im1, im2, target, city_1, city_2 in tqdm(test_dataloader):

            if args.gpu:
                im1 = im1.cuda()
                im2 = im2.cuda()
                target = target.cuda()

            ## For confusion matrix 
            y_true_temp = target.cpu().detach().numpy()
            for i in y_true_temp:
                y_true.append(i[:])    

            output_model = model_sigmoid(im1, im2)

            pred = torch.where(output_model > 0.5, 1, 0)

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

    
    if (sigmoid != True):
        model_contrastive = SiameseNetwork_contrastive.load_from_checkpoint(
            checkpoint_path=str(args.checkpoint_contrastive),
            hparams_file=str(args.hparams_contrastive),
            map_location=None,
        )
    
        logging.info(f"Loading model from : {args.checkpoint_contrastive}")
    
        model_contrastive.eval()

        if args.gpu and torch.cuda.is_available():
            model_contrastive.cuda()

        correct_euclidean = 0
        correct_cosine = 0

        cos = torch.nn.CosineSimilarity()

        for im1, im2, target, city_1, city_2 in tqdm(test_dataloader):

        # if (city_1[0] != 'Moscow'):

        #    continue

        #print(f"city 1 : {city_1} and city 2 : {city_2}")

            if args.gpu:
                im1 = im1.cuda()
                im2 = im2.cuda()
                target = target.cuda()

            output1, output2 = model_contrastive(im1, im2)

        #print(f'output1 {output1}')
        #print(f'output2 {output2}')

        #print(f'euclidean_distance shape {output1.size()}')

            euclidean_distance = torch.nn.functional.pairwise_distance(output1, output2)

        #print(f'euclidean_distance shape {euclidean_distance.size()}')
        #print(f'euclidean_distance {euclidean_distance}')

            cos_distance = cos(output1, output2)

        #print(f'cos_distance(output1, output2) {cos_distance}')

        #print(f'torch.ones_like(input) {torch.ones_like(cos_distance)}')

            cosine_distance = torch.ones_like(cos_distance) - cos_distance

        #print(f'cosine_distance {cosine_distance}')

            pred_cosine = torch.where(cosine_distance > 0.5, 1, 0)

        #print(f'pred_cosine {pred_cosine}')

            pred = torch.where(euclidean_distance > 0.5, 1, 0)

        #print(f'pred {pred}')

            correct_euclidean = correct_euclidean +  pred.eq(target.view_as(pred)).sum().item()

            correct_cosine = correct_cosine +  pred_cosine.eq(target.view_as(pred_cosine)).sum().item()

        val_acc_contrastive = 100. * correct_euclidean / dataset_length

        val_acc_cosine = 100. * correct_cosine / dataset_length

        print(f'dataset legnth : {dataset_length}')

        print('###################################')

        print(f"val_acc euclidean: {val_acc_contrastive}")

        print(f"val_acc cosine: {val_acc_cosine}")
