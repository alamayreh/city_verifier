import logging
from scipy import spatial
import pandas as pd
from PIL import Image
from argparse import  ArgumentParser
from pathlib import Path
from utils import *
from train_sigmoid import SiameseNetwork as SiameseNetwork_sigmoid
from tqdm import tqdm
from os.path import isfile, join
import math

#Rio_de_Janeiro

#("Moscow" "London" "Shanghai" "Cairo" "Delhi" "New_york" "Rio_de_Janeiro" "Sydney" "Roma" "Tokyo")

#python3 analysis_results.py --test_city New_york --database_city New_york
#python3 analysis_results.py --test_city Shanghai --database_city Shanghai
#python3 analysis_results.py --test_city Tokyo --database_city Tokyo
#python3 analysis_results.py --test_city London --database_city London
#export CUDA_VISIBLE_DEVICES=4

def parse_args():
    args = ArgumentParser()

    args.add_argument(
        "--database_csv",
        type=Path,
        default=Path("/data/omran/cities_data/dataset/cities/csv_meta/training"), 
        help="CSV folder for images database.",
    )
    
    args.add_argument(
        "--image_csv_test",
        type=Path,
        default=Path("/data/omran/cities_data/dataset/cities/csv_meta/test"),
        help="Folder containing CSV files meta data for of test images.",
    )

    args.add_argument(
        "--results_dir",
        type=Path,
        default=Path("/data/omran/cities_data/results"), 
        help="Results CSVs folder.",
    )


    args.add_argument(
        "--outlist_dir",
        type=Path,
        default=Path("/data/omran/cities_data/dataset/cities/out_list_test"), 
        help="Results CSVs folder.",
    )

    args.add_argument(
        "--image_dir_test",
        type=Path,
        default=Path("/data/omran/cities_data/dataset/cities/test"), 
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


def Probability_365(Probability_365):

  
    image_prob_str =((Probability_365.strip())[1:])[:-1].split()

    #print((image_prob_str))
    image_prob = [float(i) for i in image_prob_str]

    #image_prob= [0 if i < 0.1 else i for i in image_prob]

    return image_prob


def distance_euclidean(Probability_365_base,IMG_ID_data_base,Probability_365_test,IMG_ID_test):

    test_prob = Probability_365(Probability_365_test)
    base_prob = Probability_365(Probability_365_base)
    eDistance = math.dist((base_prob),(test_prob))
    #test_prob = Probability_365(Probability_365_test)
    #base_prob = Probability_365(Probability_365_base)

    #print("#################################################")
    # x.Probability_365_base,x.IMG_ID_data_base, x.Probability_365_test,x.IMG_ID_test
    #print(f'{IMG_ID_data_base} {base_prob}')
    ##print(f'{IMG_ID_test} {test_prob}')
    #print(eDistance)
    #print("#################################################")

    return (eDistance + 0.01)     

def distance_cos(Probability_365_test,Probability_365_base):

    test_prob = Probability_365(Probability_365_test)
    base_prob = Probability_365(Probability_365_base)

    cDistance =  1 - spatial.distance.cosine(base_prob, test_prob)

    return (cDistance + 0.01)  

if __name__ == '__main__':

    args = parse_args()

    logging.basicConfig(level=logging.INFO)

    logging.info(f"Analysis results Test {args.test_city} city on {args.database_city} database")

    # Read results 
    df = pd.read_csv(f'{args.results_dir}/{args.test_city}_on_{args.database_city}_database.csv')

    # Read similarity  
    test_simialrity_db           =  pd.read_csv(str(join(args.image_csv_test, args.test_city)) + '.csv' ,  usecols=['IMG_ID', 'S16'])
    test_simialrity_db.rename(columns={'IMG_ID': 'IMG_ID_test', 'S16': 'Probability_365_test'}, inplace=True)


    database_simialrity_db = pd.read_csv(str(join(args.database_csv, args.database_city)) + '.csv',  usecols=['IMG_ID', 'S16'])
    database_simialrity_db.rename(columns={'IMG_ID': 'IMG_ID_data_base', 'S16': 'Probability_365_base'}, inplace=True)
    df = pd.merge(df, database_simialrity_db, on="IMG_ID_data_base")
    df = pd.merge(df, test_simialrity_db,     on="IMG_ID_test")


    #Remove out list 
    #out_list = pd.read_csv(str(join(args.outlist_dir, args.test_city)) + '.csv')
    #values_list = out_list['IMG_ID'].tolist()
    #df = df[~df['IMG_ID_test'].isin(values_list)]

    
    df["votes_sigmoid"] = (np.where(df['sigmoid_output'] > 0.5,0,1))

    df["votes_diff"]    = (np.where(df['sigmoid_output'] > 0.9,1,0))
    df["votes_same"]    = (np.where(df['sigmoid_output'] < 0.1,1,0))

    df['eDistance'] = df.apply(lambda x: distance_euclidean(x.Probability_365_base,x.IMG_ID_data_base, x.Probability_365_test,x.IMG_ID_test), axis=1)
    df['cDistance'] = df.apply(lambda x: distance_cos(x.Probability_365_base, x.Probability_365_test), axis=1)

    df.drop(columns=['IMG_ID_data_base','Probability_365_base','Probability_365_test'],inplace=True)


    df['probablity_same_e'] = df.apply(lambda x:  (1 - x.sigmoid_output) * (1 / math.sqrt(x.eDistance)), axis=1)
    df['probablity_diff_e'] = df.apply(lambda x:    (  x.sigmoid_output) * (1 / math.sqrt(x.eDistance)), axis=1)

    df['probablity_same_c'] = df.apply(lambda x:  (1 - x.sigmoid_output) * (1 / math.sqrt(x.cDistance)), axis=1)
    df['probablity_diff_c'] = df.apply(lambda x:    (  x.sigmoid_output) * (1 / math.sqrt(x.cDistance)), axis=1)



    df_sum = df.groupby('IMG_ID_test').sum()

    len_images = len(df_sum.index)
    print(df_sum)

    same_db_sig      = (np.where(df_sum['votes_sigmoid'] > 800))
    same_db_sig_soft = (np.where(df_sum['sigmoid_output'] < 800))
    same_thr         = (np.where(df_sum['votes_same'] > df_sum['votes_diff'], 1, 0)).sum()

    same_db_prob    = (np.where(df_sum['probablity_same']   > df_sum['probablity_diff']))

    same_e_distance = (np.where(df_sum['probablity_same_e'] > df_sum['probablity_diff_e']))
    same_c_distance = (np.where(df_sum['probablity_same_c'] > df_sum['probablity_diff_c']))

    print((f"Analysis results Test {args.test_city} city on {args.database_city} database"))
    print("####################################################################")
    print(f"Based on votes_sigmoid hard                              : {same_db_sig[0].size / len_images}")
    print(f"Based on votes_sigmoid soft                              : {same_db_sig_soft[0].size / len_images}")
    print(f"Based on votes_sigmoid > 0.90 and < 0.01                 : {same_thr / len_images}")
    print("####################################################################")
    
   

    print(f"Based on probablity_same_similarity all 365 Euclidean    : {same_db_prob[0].size / len_images}")
    #print(f"Based on probablity_same_similarity all 365 Cosine       : {same_db_prob[0].size}")

    print(f"Based on probablity_same_similarity S16 Euclidean        : {same_e_distance[0].size  / len_images}")
    print(f"Based on probablity_same_similarity S16 Cosine           : {same_c_distance[0].size  / len_images}")
    print("####################################################################")

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

            


