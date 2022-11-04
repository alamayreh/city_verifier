from email.policy import default
import torch
import random
import logging
import torchvision
from PIL import Image
from argparse import ArgumentParser
from pathlib import Path
from utils import *
from train_triplet import SiameseNetwork
from inference import SiameseNetworkDataset
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

# export CUDA_VISIBLE_DEVICES=4,5,6,7


def parse_args():
    args = ArgumentParser()
    args.add_argument(
        "--checkpoint_triplet",
        type=Path,
        default=Path("/data/omran/cities_data/models/resnet101_64_triplet_256000/221103-1054/ckpts/epoch_29.ckpt"),
        help="Checkpoint to already trained model (*.ckpt)",
    )
    args.add_argument(
        "--hparams_triplet",
        type=Path,
        default=Path("/data/omran/cities_data/models/resnet101_64_triplet_256000/221103-1054/tb_logs/version_0/hparams.yaml"),
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
        default=24,
        help="Number of workers for image loading and pre-processing",
    )
    return args.parse_args()


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
        imageFolderDataset=DatasetFolder_test, transform=tfm_test, num_pairs=2048)

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





    logging.info(f"Loading model from : {args.checkpoint_triplet}")

    model_triplet = SiameseNetwork.load_from_checkpoint(
        checkpoint_path=str(args.checkpoint_triplet),
        hparams_file=str(args.hparams_triplet),
        map_location=None,
        )

    model_triplet.eval()

    if args.gpu and torch.cuda.is_available():
        model_triplet.cuda()

    correct = 0
    y_true = []
    y_pred = []

    cos = torch.nn.CosineSimilarity()

    for im1, im2, target, city_1, city_2 in tqdm(test_dataloader):

        if args.gpu:
            im1 = im1.cuda()
            im2 = im2.cuda()
            target = target.cuda()

        ## For confusion matrix 
        y_true_temp = target.cpu().detach().numpy()

       
        for i in y_true_temp:
            y_true.append(i[:])    

        output1, output2, _ = model_triplet(im1, im2, im2)

        cos_distance = cos(output1, output2)     
        

        cosine_distance = torch.ones_like(cos_distance) - cos_distance

        pred = torch.where(cosine_distance > 0.5, 1, 0)


        # For confusion matrix 
        y_pred_temp = pred.cpu().detach().numpy()
        #print(y_pred_temp)
        for i in y_pred_temp:
            y_pred.append(i)


        correct = correct + pred.eq(target.view_as(pred)).sum().item()

    print(confusion_matrix(y_true, y_pred))
    print(confusion_matrix(y_true, y_pred, normalize='true'))
    #print(confusion_matrix(y_true, y_pred, normalize='all'))

    print(f'acc skealrn just to check {accuracy_score(y_true, y_pred) * 100 }')
    val_acc_triplet = 100. * correct / dataset_length
    print(f"val_acc triplet:          {val_acc_triplet}")

