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

# ("Moscow" "St_Petersburg" "London" "Edinburgh" "Shanghai" "Beijing" "Cairo" "Delhi" "New_york" "Los_Angeles" "Rio_de_Janeiro" "Sydney" "Roma" "Milan" "Tokyo") 
#python3 analysis_filtered_vipp_similarity.py --test_city Los_Angeles --database_city New_york
#python3 analysis_filtered_vipp_similarity.py --test_city Shanghai --database_city Shanghai
#python3 analysis_filtered_vipp_similarity.py --test_city Tokyo --database_city Tokyo
#python3 analysis_filtered_vipp_similarity.py --test_city Moscow --database_city Moscow
#python3 analysis_filtered_vipp_similarity.py --test_city Edinburgh --database_city Edinburgh
#python3 analysis_filtered_vipp_similarity.py --test_city St_Petersburg --database_city St_Petersburg
#python3 analysis_filtered_vipp_similarity.py --test_city Milan --database_city Milan
#python3 analysis_filtered_vipp_similarity.py --test_city New_york --database_city   New_york
#export CUDA_VISIBLE_DEVICES=4


def parse_args():
    args = ArgumentParser()

    args.add_argument(
        "--S16_database",
        type=Path,
        default=Path("/data/omran/cities_data/dataset/S16_database_10k.csv"), 
        #default=Path("/data/omran/cities_data/dataset/S16_database.csv"), 
        help="CSV folder for images database.",
    )
    
    args.add_argument(
        "--vipp_database",
        type=Path,
        default=Path("/data/omran/cities_data/dataset/Vipp_classes_10k.csv"),
        #default=Path("/data/omran/cities_data/dataset/Vipp_classes.csv"),
        help="Folder containing CSV files meta data for all images.",
    )

    args.add_argument(
        "--results_dir",
        type=Path,
        default=Path("/data/omran/cities_data/results/dataset_10k/ResNet50_ImageNetT_VippTraining_test_100_restricted"),
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

def similar_images(p_base,p_test):

    test_prob = String_to_list(p_base)
    base_prob = String_to_list(p_test)

    #a_list = [10, 11, 14, 23, 9, 3, 35, 22]

    max_value_test = max(test_prob)
    max_index_test = test_prob.index(max_value_test)

    N = 1 # try to change to 1  
    res = sorted(range(len(base_prob)), key = lambda sub: base_prob[sub])[-N:]

    if (max_index_test in res ):
        out = True
    else:
        out = False    

    #cDistance =  spatial.distance.cosine(base_prob, test_prob)
    # 0 are similar, 1 are diff 

    return out


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

    # Define classes 
    #vipp_classes = {"Munich":4,"Berlin":4,"Cairo":44,"Delhi":14,"London":1,"Edinburgh":1,"Moscow":22,"St_Petersburg":22,"New_york":0,"Los_Angeles":0,"Rio_de_Janeiro":11,"Roma":5,"Milan":5,"Shanghai":10,"Beijing":10,"Sydney":8,"Tokyo":7,"Amman":58,"Istanbul":27,"Mexico_city":17,"Paris":3,"Singapore":36}    
    #vipp_classes = {"Cairo":44,"Delhi":14,"London":1,"Edinburgh":1,"Moscow":22,"St_Petersburg":22,"New_york":0,"Los_Angeles":0,"Rio_de_Janeiro":11,"Roma":5,"Milan":5,"Shanghai":10,"Beijing":10,"Sydney":8,"Tokyo":7,"Amman":58,"Istanbul":27,"Mexico_city":17,"Paris":3,"Singapore":36}    
    vipp_classes = {"Amsterdam":9,"Barcelona":2,"Florence":5,"Venice":5,"Vancouver":6,"Quebec":6,"Munich":4,"Berlin":4,"Cairo":44,"Delhi":14,"London":1,"Edinburgh":1,"Moscow":22,"St_Petersburg":22,"New_york":0,"Los_Angeles":0,"Rio_de_Janeiro":11,"Roma":5,"Rome":5,"Milan":5,"Shanghai":10,"Beijing":10,"Sydney":8,"Tokyo":7,"Amman":58,"Istanbul":27,"Mexico_city":17,"Paris":3,"Singapore":36,"NewYork":0,"LosAngeles":0}    

    print(f"Analysis results Test {args.test_city} city on {args.database_city} database, Criterion Top {Top} \n")

    # Read results from the verifier 
    df = pd.read_csv(f'{args.results_dir}/{args.test_city}_on_{args.database_city}_database.csv',usecols=['IMG_ID_test','IMG_ID_data_base','sigmoid_output'])

    # Read similarity and Vipp class
    db_16   =  pd.read_csv(args.S16_database).set_index('IMG_ID')
    db_vipp =  pd.read_csv(args.vipp_database).set_index('IMG_ID')

    
    #print(db_16.head())

    # Get the number of images from the test city 
    img_test_list = df["IMG_ID_test"].unique()
    len_images = len(img_test_list)
    print(f"Number of input images from the test city {args.test_city}      : {len_images}" )

    # Remove images from the test set, if the database city is not in one of the top 3 of the classifier output. 
    img_test_list = df["IMG_ID_test"].unique()
    out_list_database = [i for i in img_test_list if vipp_classes[args.database_city] not in get_top(db_vipp.loc[i].pred_10_classes,Top)] 
    df = df[~df['IMG_ID_test'].isin(out_list_database)]
    print(f"Number of accepted images from the test city {args.test_city} that recognized as {args.database_city} : { df['IMG_ID_test'].nunique()}" )

    df.drop(df[df['IMG_ID_test'] == df['IMG_ID_data_base']].index, inplace = True)
    #print(df.shape)

    number_voters_per_image =  df['IMG_ID_data_base'].nunique() -1 

    # number of images in database 
    print(f"Number of voters from database city {args.database_city} per image  : {number_voters_per_image}\n" )

    #print(df.shape)
    if(number_voters_per_image!=0):
        # Remove images from the database set, if the S16 class of the test images is not in one of the top N of the database image. 
        df['similar_images'] = df.apply(lambda x: similar_images(db_16.loc[x.IMG_ID_test].S16, db_16.loc[x.IMG_ID_data_base].S16), axis=1)
        #print(df)
        df.query('similar_images == True', inplace=True)


        df['cDistance'] = df.apply(lambda x: distance_cos(db_16.loc[x.IMG_ID_test].S16, db_16.loc[x.IMG_ID_data_base].S16), axis=1)
        df['probablity_same_c'] = df.apply(lambda x:  (1 - x.sigmoid_output) * ((x.cDistance)), axis=1)
        df['probablity_diff_c'] = df.apply(lambda x:    (  x.sigmoid_output) * ((x.cDistance)), axis=1)
        df['1-sigmoid_output']  = df.apply(lambda x:  (1 - x.sigmoid_output) , axis=1)
        #print(df)

    #print(df.head())
    df.drop(['IMG_ID_data_base'], axis=1,inplace=True)

    thr = 0.6

    df["votes_sigmoid_05"] = (np.where(df['sigmoid_output'] > 0.5,0,1))
    
    df["votes_diff"]       = (np.where(df['sigmoid_output'] > thr,1,0))
    df["votes_same"]       = (np.where(df['sigmoid_output'] < thr,1,0))

    #print(df)
    df_sum = df.groupby('IMG_ID_test').sum()
    #print(df_sum)

    if(number_voters_per_image!=0):
        same_c_distance = (np.where(df_sum['probablity_same_c'] > df_sum['probablity_diff_c']))
        #same_c_distance_thr = (np.where(df_sum['probablity_same_c'] > df_sum['probablity_diff_c']))

        #print(same_c_distance)
        #print(same_c_distance[0])
        #print(same_c_distance[0].size)
    #len_images = len(df_sum.index)
    #print(df_sum)
    same_db_sig      = (np.where(df_sum['votes_sigmoid_05'] > int(number_voters_per_image / 2 )))
    same_db_sig_soft = (np.where(df_sum['sigmoid_output'] < df_sum['1-sigmoid_output'], 1, 0)).sum()  #(np.where(df_sum['sigmoid_output']   <  df['1-sigmoid_output'] ))

    same_thr         = (np.where(df_sum['votes_same'] > df_sum['votes_diff'], 1, 0)).sum()

    print("Only Similar images vote! Top1 from S16\n")

    
    print(f"Based on votes_sigmoid hard 0.5 thr                   : {same_db_sig[0].size / len_images}")
    print(f"Based on votes_sigmoid soft                           : {same_db_sig_soft / len_images}")
    print(f"Based on votes_sigmoid > {thr} and < {thr}                : {same_thr / len_images}")
    #print("--------------------------------------------------------------------")
    if(number_voters_per_image!=0):
        print(f"Based on probablity_same_similarity S16 Cosine        : {same_c_distance[0].size  / len_images}")

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


