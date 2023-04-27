'''
This script calculates verification accuracy defined in Equation 2.
'''

import logging
from scipy import spatial
import pandas as pd
from PIL import Image
from argparse import  ArgumentParser
from pathlib import Path
from tqdm import tqdm
from os.path import isfile, join
import math
import numpy as np


#python3 verification_acc.py --set open  --thr 0.6 --GeoVIPP --print_details --Similarity
#python3 verification_acc.py --set closed  --thr 0.6 --GeoVIPP --print_details --Similarity

#python3 verification_acc.py --set open  --thr 0.6 --GeoVIPP --print_details 


def parse_args():
    args = ArgumentParser()

    args.add_argument(
        "--S16_database",
        type=Path,
        default=Path("/data/omran/cities_data/dataset/S16_database_10k.csv"), 
        help="CSV folder for images database.",
    )
    
    args.add_argument(
        "--vipp_database",
        type=Path,
        default=Path("/data/omran/cities_data/dataset/Vipp_classes_10k.csv"),
        help="Folder containing CSV files meta data for all images.",
    )

    args.add_argument(
        "--results_dir",
        type=Path,
        
        #default=Path("/data/omran/cities_data/results/dataset_10k/ViT_ImgNet/Base"),

        #default=Path("/data/omran/cities_data/results/dataset_10k/ViT_ImgNet/GeoVIPP_100"),
        #default=Path("/data/omran/cities_data/results/dataset_10k/ViT_ImgNet/Similarity_025"),
        default=Path("/data/omran/cities_data/results/dataset_10k/ViT_ImgNet/GeoVipp_50"),
        #default=Path("/data/omran/cities_data/results/dataset_10k/ViT_ImgNet/GeoVIPP_25_similarity_25"),

        help="Results CSVs folder.",
    )

    args.add_argument(
        "--set",
        type=str,
        help="Open or closed set",
    )

    args.add_argument(
        "--Similarity",
        action="store_true",
    )
    args.add_argument(
        "--GeoVIPP",
        action="store_true",
    )

    args.add_argument(
        "--print_details",
        action="store_true",
        help="Use GPU for inference if CUDA is available",
    )
    args.add_argument(
        "--thr",
        type=float,
        default=0.6,
        help="Test",
    )

    return args.parse_args()


def String_to_list(in_string):

  
    image_prob_str =((in_string.strip())[1:])[:-1].split()


    image_prob = [float(i) for i in image_prob_str]


    return image_prob


def distance_euclidean(p_base,p_test):

    test_prob = String_to_list(p_test)
    base_prob = String_to_list(p_base)
    eDistance = math.dist((base_prob),(test_prob))

    return (eDistance )     


def get_top(top10,num):

    #print(top10)
    image_prob_str = ((top10.strip())[1:])[:-1].split()
    list10 = [int(i) for i in image_prob_str]

    return list10[0:num]    

def distance_cos(p_base,p_test):

    test_prob = String_to_list(p_base)
    base_prob = String_to_list(p_test)

    cDistance =  spatial.distance.cosine(base_prob, test_prob)
    # 0 are similar, 1 are diff 

    return (cDistance)  


if __name__ == '__main__':

    Top = 2

    args = parse_args()

    logging.basicConfig(level=logging.INFO)

    thr = args.thr

    vipp_classes = {"Amsterdam":9,"Barcelona":2,"Florence":5,"Venice":5,"Vancouver":6,"Quebec":6,"Munich":4,"Berlin":4,"Cairo":44,"Delhi":14,"London":1,"Edinburgh":1,"Moscow":22,"St_Petersburg":22,"New_york":0,"Los_Angeles":0,"Rio_de_Janeiro":11,"Roma":5,"Rome":5,"Milan":5,"Shanghai":10,"Beijing":10,"Sydney":8,"Tokyo":7,"Amman":58,"Istanbul":27,"Mexico_city":17,"Paris":3,"Singapore":36,"NewYork":0,"LosAngeles":0}    

    if(args.set == 'closed'):
        city_list = ["Amsterdam", "Barcelona", "Berlin", "London", "NewYork", "LosAngeles", "Rome", "Milan", "Paris", "Tokyo"]
        #city_list = ["NewYork", "LosAngeles", "Rome", "Milan"]

    if(args.set == 'open'):
        city_list = ["Amman", "Istanbul", "Mexico_city", "Singapore", "Quebec", "Vancouver","Florence", "Rome", "St_Petersburg","Edinburgh"]
        #city_list = ["Amman", "Istanbul", "Mexico_city", "Singapore", "Quebec", "Vancouver", "Venice", "Florence", "Rome", "Moscow", "St_Petersburg", "Shanghai", "Beijing", "London", "Edinburgh"]
        #city_list = ["Quebec", "Vancouver", "Venice", "Florence", "Rome", "Moscow", "St_Petersburg", "Shanghai", "Beijing", "London", "Edinburgh"]

    N = len(city_list)

    # Read similarity  
    db_16   =  pd.read_csv(args.S16_database).set_index('IMG_ID')
    db_vipp =  pd.read_csv(args.vipp_database).set_index('IMG_ID')

    diag_hard = 0 
    off_diag_hard = 0    
    
    diag_soft = 0
    off_diag_soft = 0
    
    diag_thr = 0
    off_diag_thr  = 0
    
    diag_similarity   = 0
    off_diag_similarity   = 0
    
    for test_city in city_list:
        for database_city in city_list:

            if(args.print_details):
                print(f"Analysis results Test {test_city} city on {database_city} database.\n")

            # Read results 
            df = pd.read_csv(f'{args.results_dir}/{test_city}_on_{database_city}_database.csv',usecols=['IMG_ID_test','IMG_ID_data_base','sigmoid_output'])
            
            img_test_list = df["IMG_ID_test"].unique()
            len_images = len(img_test_list)
            #if(args.print_details):
            #    print(f"Number of input images from the test city {test_city}      : {len_images}" )

            if(args.GeoVIPP):

                # Remove images from the test set, if the database city is not in on of the top 3 of the classifier output.
                img_test_list = df["IMG_ID_test"].unique()
                out_list_database = [i for i in img_test_list if vipp_classes[database_city] not in get_top(db_vipp.loc[i].pred_10_classes,Top)] 
                df = df[~df['IMG_ID_test'].isin(out_list_database)]
                if(args.print_details):
                    print(f"Number of accepted images from the test city {test_city} that recognized as {database_city} : { df['IMG_ID_test'].nunique()}\n" )

            if(df['IMG_ID_test'].nunique()==0):
                if(args.print_details):
                    print(f'results : 0.00')

                votes_sigmoid_hard = 0
                votes_sigmoid_soft = 0
                votes_sigmoid_thr  = 0 
                votes_similarity   = 0

                continue
            # remove the same image from the database 
            df.drop(df[df['IMG_ID_test'] == df['IMG_ID_data_base']].index, inplace = True)
            
            number_voters_per_image =  df['IMG_ID_data_base'].nunique() - 1
            #if(args.print_details):            
            #    print(f"Number of voters from database city {database_city} per image  : {number_voters_per_image}\n" )

            if(number_voters_per_image > 0):
                df['cDistance'] = df.apply(lambda x: distance_cos(db_16.loc[x.IMG_ID_test].S16, db_16.loc[x.IMG_ID_data_base].S16), axis=1)
                df['probablity_same_c'] = df.apply(lambda x:    (1-x.sigmoid_output) * (1-(x.cDistance)), axis=1)
                df['probablity_diff_c'] = df.apply(lambda x:    (x.sigmoid_output) *   (1-(x.cDistance)), axis=1)

            #print(df)
            if(args.Similarity):
                df = df.loc[df['cDistance'] < 0.25]

            #print(df)

            df.drop(['IMG_ID_data_base'], axis=1,inplace=True)


            df["votes_sigmoid_05_diff"] = (np.where(df['sigmoid_output'] >  0.5,1,0))
            df["votes_sigmoid_05_same"] = (np.where(df['sigmoid_output'] <= 0.5,1,0))


            df["votes_diff_thr"]       = (np.where(df['sigmoid_output'] >  thr,1,0))
            df["votes_same_thr"]       = (np.where(df['sigmoid_output'] <= thr,1,0))

            df["1_sigmoid_output"] = 1 - df['sigmoid_output']


            df_sum = df.groupby('IMG_ID_test').sum()

            #print(df_sum)





            same_db_sig      = (np.where(df_sum['votes_sigmoid_05_same'] > df_sum['votes_sigmoid_05_diff'], 1, 0)).sum()


            same_db_sig_soft = (np.where(df_sum['sigmoid_output'] < df_sum['1_sigmoid_output'], 1, 0)).sum()

            same_thr         = (np.where(df_sum['votes_same_thr'] > df_sum['votes_diff_thr'], 1, 0)).sum()

            same_c_distance = (np.where(df_sum['probablity_same_c'] > df_sum['probablity_diff_c'], 1, 0)).sum()

            votes_sigmoid_hard = same_db_sig / len_images

            votes_sigmoid_soft = same_db_sig_soft/ len_images
            votes_sigmoid_thr  = same_thr / len_images

            votes_similarity   = same_c_distance  / len_images
            
  
            if(test_city==database_city):
                    diag_hard+=votes_sigmoid_hard
                    diag_soft+=votes_sigmoid_soft
                    diag_thr +=votes_sigmoid_thr
                    diag_similarity+=votes_similarity
            else:
                    off_diag_hard+=votes_sigmoid_hard
                    off_diag_soft+=votes_sigmoid_soft
                    off_diag_thr +=votes_sigmoid_thr
                    off_diag_similarity+=votes_similarity

            if(args.print_details):
                #print(f"Based on votes_sigmoid hard 0.5 thr                      : {votes_sigmoid_hard}")
                #print(f"Based on votes_sigmoid soft                              : {votes_sigmoid_soft}")
                #print(f"Based on votes_sigmoid > {thr} and < {thr}                   : {votes_sigmoid_thr}")
                print("--------------------------------------------------------------------")           
                print(f"Based on probablity_same_similarity S16 Cosine           : {votes_similarity}")
                print("--------------------------------------------------------------------")

            #print(df_cosine_votes)


            #print(df_cosine_votes)        

        if(args.print_details):
            print("####################################################################")

    print(f'Results path {args.results_dir}\n')
    print(f'Number of cities : {N}, and the city list is {city_list}\n')    
    
    print(f'sigmoid hard 0.5 thr; acc : {0.5 * diag_hard/N + 0.5 * (1 - off_diag_hard /(N * N - N))},   Diagonal : {diag_hard/N} and off-diagonal as {off_diag_hard /(N * N - N)}')
    print(f'sigmoid soft        ; acc : {0.5 * diag_soft/N + 0.5 * (1 - off_diag_soft /(N * N - N))},  Diagonal : {diag_soft/N} and off-diagonal as {off_diag_soft /(N * N - N) }')
    print(f'sigmoid thr {thr}     ; acc : {0.5 * diag_thr/N  + 0.5 * (1 - off_diag_thr /(N * N - N)) },  Diagonal : {diag_thr/N} and off-diagonal as {off_diag_thr /(N * N - N)}')
    print(f'similarity cosine   ; acc : {0.5 * diag_similarity/N + 0.5 * (1 - off_diag_similarity/(N * N - N))},  Diagonal : {diag_similarity/N} and off-diagonal as {off_diag_similarity/(N * N - N)}')
    print("####################################################################")

