import numpy as np
import pandas as pd
import os
from os.path import isfile, join
import math
from scipy import spatial
import shutil

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

out_list = [0, 1, 2, 24, 42, 58, 69, 76, 86, 149, 164, 171, 174, 196,
            207, 243, 247, 275, 276, 293, 306, 310, 312, 313, 314, 341,
            351,197,364,345,62,83,43,131,202,179,210,290,191,304,165]

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

def filter_and_copy(IMG_ID, Prob_365, S16,in_images_city,out_images_city):

    satisfy = True
    prob_365 = string_to_prob(Prob_365)
    prob_S16 = string_to_prob(S16)

    idx_365 = sorted(range(len(prob_365)), key=lambda i: prob_365[i])[-10:]

    if (idx_365[9] in out_list or idx_365[8] in out_list or idx_365[7] in out_list ):
        satisfy = False

    # vote for the indoor or outdoor
    io_image = np.mean(labels_IO[idx_365[:10]])
    if io_image < 0.98:
        in_out = 'indoor'
        satisfy = False
    else:
        in_out = 'outdoor'

    indoor_16 = np.sum(prob_S16[0:5])
    nature_16 = np.sum(prob_S16[6:9])
    urban_16  = np.sum(prob_S16[10:])

    if (indoor_16 > nature_16 and indoor_16 > urban_16):
        out_in_16 = 'indoor_16'
        satisfy = False
    elif (nature_16 > indoor_16 and nature_16 > urban_16):
        out_in_16 = 'nature_16'
        satisfy = False
    elif (urban_16 > indoor_16 and urban_16 > nature_16):
        out_in_16 = 'urban_16'

    if(satisfy):
        # do the copy 

        shutil.copyfile(join(in_images_city,IMG_ID),join(out_images_city,IMG_ID))

    return 0


#meta_folder_path = '/data/omran/cities_data/dataset/cities/csv_meta/training'
#folder_in_images = '/data/omran/cities_data/dataset/cities/training'
#folder_out_images= '/data/omran/cities_data/dataset/filtered/training'

#meta_folder_path = '/data/omran/cities_data/dataset/cities/csv_meta/open_set'
#folder_in_images = '/data/omran/cities_data/dataset/open_set'
#folder_out_images= '/data/omran/cities_data/dataset/filtered/open_set'

meta_folder_path = '/data/omran/cities_data/dataset/cities/csv_meta/new_batch_cities'

folder_in_images = '/data/omran/cities_data/dataset/cities/new_batch_cities/'
folder_out_images= '/data/omran/cities_data/dataset/filtered/new_batch_cities'

    
#for city in os.listdir(meta_folder_path):
for city in ['Vancouver.csv']:


    df = pd.read_csv(join(meta_folder_path, city))
    in_images_city = join(folder_in_images, city[:-4])
    out_images_city= join(folder_out_images,city[:-4])

    if not os.path.exists(out_images_city):
        os.makedirs(out_images_city)
    print(out_images_city)
    
    #print(df.head())
    df.apply(lambda x:  filter_and_copy(x.IMG_ID, x.Probabily_365, x.S16,in_images_city,out_images_city), axis=1)


#df = pd.read_csv(join(meta_folder_path, city))
#input_image = df.query('IMG_ID == "2674294937.jpeg"').reset_index(drop=True)
#print(input_image)
#image_class(input_image.iloc[0].IMG_ID, input_image.iloc[0].Probabily_365,input_image.iloc[0].S16, city)

#df.apply(lambda x:  get_similar_images(input_image.iloc[0].IMG_ID, input_image.iloc[0].Probabily_365, input_image.iloc[0].S16, x.IMG_ID, x.Probabily_365, x.S16, city), axis=1)



#db.to_csv(join(meta_folder_path, i), index=False)
# print(db.head())
