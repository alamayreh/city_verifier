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
#Rio_de_Janeiro

#("Moscow" "London" "Shanghai" "Cairo" "Delhi" "New_york" "Rio_de_Janeiro" "Sydney" "Roma" "Tokyo")

#python3 analysis_filtered_vipp.py --test_city New_york --database_city New_york
#python3 analysis_filtered_vipp.py --test_city Shanghai --database_city Shanghai
#python3 analysis_filtered_vipp.py --test_city Tokyo --database_city Tokyo
#python3 analysis_filtered_vipp.py --test_city Moscow --database_city Moscow
#export CUDA_VISIBLE_DEVICES=4

def parse_args():
    args = ArgumentParser()

    args.add_argument(
        "--S16_database",
        type=Path,
        default=Path("/data/omran/cities_data/dataset/S16_database.csv"), 
        #default=Path("/data/omran/cities_data/dataset/S16_database_open_set.csv"), 
        help="CSV folder for images database.",
    )
    
    args.add_argument(
        "--vipp_database",
        type=Path,
        default=Path("/data/omran/cities_data/dataset/Vipp_classes.csv"),
        #default=Path("/data/omran/cities_data/dataset/Vipp_classes_open_set.csv"),
        help="Folder containing CSV files meta data for all images.",
    )

    args.add_argument(
        "--results_dir",
        type=Path,
        default=Path("/data/omran/cities_data/results/sigmoid_filtered_datast"), 
        #default=Path("/data/omran/cities_data/results/sigmoid_filtered_datast_open_set"), 
        help="Results CSVs folder.",
    )

    args.add_argument(
        "--database_city",
        type=str,
        help="Database city",
    )

    args.add_argument(
        "--test_city",
        type=str,
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

def distance_cos(p_base,p_test):

    test_prob = String_to_list(p_base)
    base_prob = String_to_list(p_test)

    cDistance =  spatial.distance.cosine(base_prob, test_prob)
    # 0 are similar, 1 are diff 

    return (1 - cDistance)  

def add_jpeg_suffix(in_string):
    return (in_string + '.jpeg')

def get_top(top10,num):

    image_prob_str = ((top10.strip())[1:])[:-1].split()
    list10 = [int(i) for i in image_prob_str]

    return list10[0:num]    

if __name__ == '__main__':

    Top = 3
    args = parse_args()

    logging.basicConfig(level=logging.INFO)

    vipp_classes = {"Cairo":44,"Delhi":14,"London":1,"Moscow":22,"New_york":0,"Rio_de_Janeiro":11,"Roma":5,"Shanghai":10,"Sydney":8,"Tokyo":7,"Amman":58,"Istanbul":27,"Mexico_city":17,"Paris":3,"Singapore":36}    
    print(f"Analysis results Test {args.test_city} city on {args.database_city} database, Criterion Top {Top} \n")

    # Read results 
    df = pd.read_csv(f'{args.results_dir}/{args.test_city}_on_{args.database_city}_database.csv',usecols=['IMG_ID_test','IMG_ID_data_base','sigmoid_output'])

    # Read similarity  
    db_16   =  pd.read_csv(args.S16_database).set_index('IMG_ID')
    db_vipp =  pd.read_csv(args.vipp_database).set_index('IMG_ID')

    # Remove images from the test set, if the classifier does not recognize the country of the test city.
    img_test_list = df["IMG_ID_test"].unique()
    len_images = len(img_test_list)
    print(f"Number of input images from the test city {args.test_city}      : {len_images}" )

    #out_list = [i for i in img_test_list if vipp_classes[args.test_city] not in get_top(db_vipp.loc[i].pred_10_classes,Top)]
    #df = df[~df['IMG_ID_test'].isin(out_list)]
    #print(f"Number of accepted images from the test city {args.test_city}   : { df['IMG_ID_test'].nunique()}" )

    # Remove images from the test set, if the database city is not in on of the top 5 of the classifier output.
    img_test_list = df["IMG_ID_test"].unique()
    out_list_database = [i for i in img_test_list if vipp_classes[args.database_city] not in get_top(db_vipp.loc[i].pred_10_classes,Top)]
   
    df = df[~df['IMG_ID_test'].isin(out_list_database)]

    print(f"Number of accepted images from the test city {args.test_city} that recognized as {args.database_city} : { df['IMG_ID_test'].nunique()}" )
    #print(df.head())

    number_voters_per_image =  df['IMG_ID_data_base'].nunique()
    


    if(number_voters_per_image!=0):
        df['cDistance'] = df.apply(lambda x: distance_cos(db_16.loc[x.IMG_ID_test].S16, db_16.loc[x.IMG_ID_data_base].S16), axis=1)
        df['probablity_same_c'] = df.apply(lambda x:  (1 - x.sigmoid_output) * ((x.cDistance)), axis=1)
        df['probablity_diff_c'] = df.apply(lambda x:    (  x.sigmoid_output) * ((x.cDistance)), axis=1)


    #df.query('cDistance >= 0.5', inplace=True)
    #print(df.head())
    print(f"Number of voters from database city {args.database_city} per image  : {number_voters_per_image}\n" )

    #print(df.head())
    df.drop(['IMG_ID_data_base'], axis=1,inplace=True)

    
    thr = 0.85

    df["votes_sigmoid_05"] = (np.where(df['sigmoid_output'] > 0.5,0,1))
    df["votes_diff"]       = (np.where(df['sigmoid_output'] > thr,1,0))
    df["votes_same"]       = (np.where(df['sigmoid_output'] < thr,1,0))

    df_sum = df.groupby('IMG_ID_test').sum()
    if(number_voters_per_image!=0):
        same_c_distance = (np.where(df_sum['probablity_same_c'] > df_sum['probablity_diff_c']))
        #same_c_distance_thr = (np.where(df_sum['probablity_same_c'] > df_sum['probablity_diff_c']))

        #print(same_c_distance)
        #print(same_c_distance[0])
        #print(same_c_distance[0].size)
    #len_images = len(df_sum.index)
    #print(df_sum)
    same_db_sig      = (np.where(df_sum['votes_sigmoid_05'] > int(number_voters_per_image / 2 )))
    same_db_sig_soft = (np.where(df_sum['sigmoid_output']   <  int(number_voters_per_image / 2 )))
    same_thr         = (np.where(df_sum['votes_same'] > df_sum['votes_diff'], 1, 0)).sum()

    
    #print(f"Based on votes_sigmoid hard 0.5 thr                      : {same_db_sig[0].size / len_images}")
    #print(f"Based on votes_sigmoid soft                              : {same_db_sig_soft[0].size / len_images}")
    print(f"Based on votes_sigmoid > {thr} and < {thr}                 : {same_thr / len_images}")
    #print("--------------------------------------------------------------------")
    if(number_voters_per_image!=0):
        print(f"Based on probablity_same_similarity S16 Cosine           : {same_c_distance[0].size  / len_images}")

    print("####################################################################")
    
   




    #df['probablity_same_e'] = df.apply(lambda x:  (1 - x.sigmoid_output) * (1 / (x.eDistance)), axis=1)
    #df['probablity_diff_e'] = df.apply(lambda x:    (  x.sigmoid_output) * (1 / (x.eDistance)), axis=1)

    #df['probablity_same_c'] = df.apply(lambda x:  (1 - x.sigmoid_output) * (1 / (x.cDistance)), axis=1)
    #df['probablity_diff_c'] = df.apply(lambda x:    (  x.sigmoid_output) * (1 / (x.cDistance)), axis=1)


    #df['eDistance'] = df.apply(lambda x: distance_euclidean(x.Probability_365_base,x.IMG_ID_data_base, x.Probability_365_test,x.IMG_ID_test), axis=1)
    #df['cDistance'] = df.apply(lambda x: distance_cos(x.Probability_365_base, x.Probability_365_test), axis=1)


    #print(f"Based on probablity_same_similarity all 365 Euclidean    : {same_db_prob[0].size / len_images}")
    #print(f"Based on probablity_same_similarity all 365 Cosine       : {same_db_prob[0].size}")

    #print(f"Based on probablity_same_similarity S16 Euclidean        : {same_e_distance[0].size  / len_images}")
    #print(f"Based on probablity_same_similarity S16 Cosine           : {same_c_distance[0].size  / len_images}")
    #print("####################################################################")

    #test_dir      = join(args.image_dir_test, args.test_city)
    #print("Images belongs to the dataste : ")
    #c = 1 
    #for index, row in tqdm(df_sum.iterrows(),  total=(df_sum.shape[0])):
    #    if(row.probablity_same > row.probablity_diff):
    #        image_path = str(join(test_dir, index)) 
    #        print(f'{c} : {image_path}')
    #        c+=1
    #print("Images does not belong to the dataste : ")
    #c = 1 
    #for index, row in tqdm(df_sum.iterrows(),  total=(df_sum.shape[0])):           
    #    if(row.probablity_same < row.probablity_diff):
    #        image_path = str(join(test_dir, index)) 
    #        print(f'{c} : {image_path}')
    #        c+=1

            


    #same_e_distance = (np.where(df_sum['probablity_same_e'] > df_sum['probablity_diff_e']))
    #same_c_distance = (np.where(df_sum['probablity_same_c'] > df_sum['probablity_diff_c']))


