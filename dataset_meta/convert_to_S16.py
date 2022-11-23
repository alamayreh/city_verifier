import numpy as np
import pandas as pd
import os 
from os.path import isfile, join


scene_matrix  =  (pd.read_excel("/data/omran/cities_data/places365_model/Scene_hierarchy.xlsx",index_col=None)).to_numpy()

print(scene_matrix)

def Probability_365(Probability_365):

    image_prob_str =((Probability_365)[1:])[:-1].split(' ')
    image_prob = [float(i) for i in image_prob_str]

    #image_prob= [0 if i < 0.01 else i for i in image_prob]

    return image_prob


def convert_to_S16(input_prob,scene_matrix,IMG_ID,city ):

    input_prob = Probability_365(input_prob)

    input_prob = np.array([input_prob]) 
    Probability_16  = np.dot(input_prob , scene_matrix)
    Probability_16 = Probability_16[0]
    
    #s16 = Probability_16
    s16 = Probability_16 / np.linalg.norm(Probability_16, ord=1)

    if (np.sum(s16[0:10]) > np.sum(s16[10:])):
        print(f"We have got a problem here /data/omran/cities_data/dataset/cities/validation/{city[:-4]}/{IMG_ID}")

    return s16



folder_path = "/data/omran/cities_data/dataset/cities/csv_meta/validation"

for i in os.listdir(folder_path):



    db           =  pd.read_csv(join(folder_path, i))

    print(f'city {i}')

    db['S16'] = db.apply(lambda x:  convert_to_S16(x.Probabily_365,scene_matrix,x.IMG_ID,i), axis=1)

    db.to_csv(join(folder_path, i), index=False)


    #print(db.head())    