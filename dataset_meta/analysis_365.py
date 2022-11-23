import numpy as np
import pandas as pd
import os
from os.path import isfile, join
import math
from scipy import spatial

scene_matrix = (pd.read_excel(
    "/data/omran/cities_data/places365_model/Scene_hierarchy.xlsx", index_col=None)).to_numpy()
S16_categories = pd.read_excel(
    "/data/omran/cities_data/places365_model/Scene_hierarchy.xlsx").columns.values.tolist()

S16_categories = tuple(S16_categories)

file_name_IO = '/data/omran/cities_data/places365_model/IO_places365.txt'
with open(file_name_IO) as f:
    lines = f.readlines()
    labels_IO = []
    for line in lines:
        items = line.rstrip().split()
        labels_IO.append(int(items[-1]) - 1)  # 0 is indoor, 1 is outdoor
labels_IO = np.array(labels_IO)

classes = list()
file_name_category = "/data/omran/cities_data/places365_model/categories_places365.txt"
with open(file_name_category) as class_file:
    for line in class_file:
        classes.append(line.strip().split(' ')[0][3:])
classes = tuple(classes)


def string_to_prob(Probability_365):

    image_prob_str = ((Probability_365)[1:])[:-1].split()
    image_prob = [float(i) for i in image_prob_str]

    return image_prob

def distance_euclidean(prob_0,prob_1):

    #test_prob = string_to_prob(prob_0)
    #base_prob = string_to_prob(prob_1)
    eDistance = math.dist((prob_0),(prob_1))

    return (eDistance)

def distance_cos(prob_0,prob_1):

    cDistance = spatial.distance.cosine(prob_0, prob_1)

    return (cDistance)

def image_class(IMG_ID, Prob_365, S16, city):

    prob_365 = string_to_prob(Prob_365)
    prob_S16 = string_to_prob(S16)

    idx_365 = sorted(range(len(prob_365)), key=lambda i: prob_365[i])[-10:]
    s16_idx = sorted(range(len(prob_S16)), key=lambda i: prob_S16[i])[-10:]

    # vote for the indoor or outdoor
    io_image = np.mean(labels_IO[idx_365[:10]])
    if io_image < 0.75:
        in_out = 'indoor'
    else:
        in_out = 'outdoor'

    indoor_16 = np.sum(prob_S16[0:5])
    nature_16 = np.sum(prob_S16[6:9])
    urban_16  = np.sum(prob_S16[10:])

    if (indoor_16 > nature_16 and indoor_16 > urban_16):
        out_in_16 = 'indoor_16'
    elif (nature_16 > indoor_16 and nature_16 > urban_16):
        out_in_16 = 'nature_16'
    elif (urban_16 > indoor_16 and urban_16 > nature_16):
        out_in_16 = 'urban_16'

    #print(f"{in_out} /data/omran/cities_data/dataset/cities/test/{city[:-4]}/{IMG_ID}")

    print(
        f"{in_out} {out_in_16} /data/omran/cities_data/dataset/cities/test/{city[:-4]}/{IMG_ID}")

    for i in range(0, 10):
        print(
            '{:.3f} -> {}'.format(float(prob_365[idx_365[i]]), classes[idx_365[i]]))
    print('----------------------------------------------------------------------------------------')

    for i in range(0, 10):
        print(
            '{:.3f} -> {}'.format(float(prob_S16[s16_idx[i]]), S16_categories[s16_idx[i]]))
    print('########################################################################################')
    return 0

def get_similar_images(IMG_ID_0, Prob_365_0, S16_0, IMG_ID_1, Prob_365_1, S16_1, city):

    
    prob_365_0 = string_to_prob(Prob_365_0)
    prob_365_1 = string_to_prob(Prob_365_1)

    prob_16_0 = string_to_prob(S16_0)
    prob_16_1 = string_to_prob(S16_1)

    idx_365_0  = sorted(range(len(prob_365_0)), key=lambda i: prob_365_0[i])[-10:]
    idx_365_1  = sorted(range(len(prob_365_1)), key=lambda i: prob_365_1[i])[-10:]
    prob_365_0 = [x if i in idx_365_0 else 0  for i,x in enumerate(prob_365_0)] 
    prob_365_1 = [x if i in idx_365_1 else 0  for i,x in enumerate(prob_365_1)] 
    

    e365 = distance_euclidean(prob_365_0,prob_365_1)
    c365 = distance_cos(prob_365_0,prob_365_1)

    e16  = distance_euclidean(prob_16_0,prob_16_1)
    c16  = distance_cos(prob_16_0,prob_16_1)


    print(f"/data/omran/cities_data/dataset/cities/test/{city[:-4]}/{IMG_ID_1}")
    print("Euclidean_365 : {:.3f} | Cosine_365 {:.3f} | ".format(e365,c365))
    print("Euclidean_16  : {:.3f} | Cosine_16  {:.3f} | ".format(e16,c16))
    print('----------------------------------------------------------------------------------------')

    return 0


#8Imf_kOVFLA.jpeg , 5558404144.jpeg
# /data/omran/cities_data/dataset/cities/test/Cairo/48839637632.jpeg
# /data/omran/cities_data/dataset/cities/test/Cairo/331164778439659.jpeg
folder_path = "/data/omran/cities_data/dataset/cities/csv_meta/test"
city = 'Moscow.csv'
#print(f'city {city}')

df = pd.read_csv(join(folder_path, city))
input_image = df.query('IMG_ID == "2674294937.jpeg"').reset_index(drop=True)
#print(input_image)
image_class(input_image.iloc[0].IMG_ID, input_image.iloc[0].Probabily_365,input_image.iloc[0].S16, city)

df.apply(lambda x:  get_similar_images(input_image.iloc[0].IMG_ID, input_image.iloc[0].Probabily_365, input_image.iloc[0].S16, x.IMG_ID, x.Probabily_365, x.S16, city), axis=1)

#db.to_csv(join(folder_path, i), index=False)
# print(db.head())
