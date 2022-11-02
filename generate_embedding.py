"""
This script takes a model and folder of images and returns a tensor that contains the embedding of the images  
"""
import torch
import logging
import torchvision
from argparse import  ArgumentParser
from pathlib import Path
from utils import *
from train_contrastive import SiameseNetwork as SiameseNetwork_contrastive
from tqdm import tqdm

#export CUDA_VISIBLE_DEVICES=4,5,6,7

def parse_args():
    args = ArgumentParser()
    args.add_argument(
        "--checkpoint_contrastive",
        type=Path,
        default=Path("/data/omran/cities_data/models/resnet101_128_sigmoid_acc_pretrain_ImgNet/220916-0358/ckpts/epoch_621.ckpt"),
        help="Checkpoint to already trained model (*.ckpt)",
    )
    args.add_argument(
        "--hparams_contrastive",
        type=Path,
        default=Path("/data/omran/cities_data/models/resnet101_128_sigmoid_acc_pretrain_ImgNet/220916-0358/tb_logs/version_0/hparams.yaml"),
        help="Path to hparams file (*.yaml) generated during training",
    )
    args.add_argument(
        "--image_dir",
        type=Path,
        default=Path("/data/omran/cities_data/dataset/cities/test"),
        help="Folder containing test set images.",
    )
    
    args.add_argument(
        "--out_dir",
        type=Path,
        default=Path("/data/omran/cities_data/embeddings/test_no_VIPP"),
        help="Folder containing output embedding.",
    )
    # environment
    args.add_argument(
        "--gpu",
        action="store_false",
        help="Use GPU for inference if CUDA is available",
    )
    args.add_argument("--batch_size", type=int, default=1)
    args.add_argument(
        "--num_workers",
        type=int,
        default=1,
        help="Number of workers for image loading and pre-processing",
    )
    return args.parse_args()


def dataloader_one_class(image_dir, batch_size, num_workers):


    tfm_test = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize(224),
            torchvision.transforms.CenterCrop(224),
            torchvision.transforms.ToTensor(),
        ]
    )

    dataset = torchvision.datasets.ImageFolder(image_dir,transform=tfm_test)

    dict = dataset.class_to_idx 
    dict_class = {v: k for k, v in dict.items()}
    logging.info(f"Class dictionary : {dict_class}")
    # {'Cairo': 0, 'Delhi': 1, 'London': 2, 'Moscow': 3, 'New_york': 4, 'Rio_de_Janeiro': 5, 'Roma': 6, 'Shanghai': 7, 'Sydney': 8, 'Tokyo': 9} 

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        shuffle=False
    )

    return dataloader,dict_class


if __name__ == '__main__':

    args = parse_args()

    logging.basicConfig(level=logging.INFO)

    logging.info(f"Loading test data : {args.image_dir}")

    dataloader_one_class, dict_class = dataloader_one_class(args.image_dir, args.batch_size, args.num_workers)

    dataset_length = len(dataloader_one_class.dataset)
    logging.info(f"Number of images: {dataset_length}")

    if len(dataloader_one_class.dataset) == 0:
        raise RuntimeError(f"No images found in {args.image_dir}")

    logging.info(f"Loading model from : {args.checkpoint_contrastive}")

    model_contrastive = SiameseNetwork_contrastive.load_from_checkpoint(
        checkpoint_path=str(args.checkpoint_contrastive),
        hparams_file=str(args.hparams_contrastive),
        map_location=None,
    )
    
    if args.gpu and torch.cuda.is_available():
        model_contrastive.cuda()

    model_contrastive.eval()

    #first_image, target_last = next(iter(dataloader_one_class)) 
    flag_first = True

    #print(class_tensors)
    #print('target_last',target_last)
    
    for im,target in tqdm(dataloader_one_class):

        if args.gpu:
            im = im.cuda()
            target = target.cuda()    

        target_next = target

        out_model, _ = model_contrastive(im, im)
        output = out_model.cpu().detach().numpy()

        

        if(flag_first):

            target_last  = target
            out_list     = output
            flag_first   = False

        else:          

            if(target_next != target_last):

                target_num = (target_last.cpu().detach().numpy()[0])

                np.save( str(args.out_dir) + f'/{dict_class[target_num]}_{target_num}', out_list) # save

                target_last = target
                out_list    = output

            else:

                out_list =  np.concatenate((out_list, output), axis=0)
                target_last = target_next
        
 
    np.save( str(args.out_dir) + f'/{dict_class[target_num]}_{target_num}', out_list) # save