"Prepare meta data; train, validation, and test"
import os
import pandas as pd
import numpy as np 
from tqdm import tqdm
import threading
import pandas as pd
from datetime import datetime
from multiprocessing import Pool


#df1 = pd.read_csv('/data/omran/cities_data/dataset/Flickr.csv' ,   usecols=['IMG_ID','Top5classes'])
#df2 = pd.read_csv('/data/omran/cities_data/dataset/Splash.csv' ,   usecols=['IMG_ID','Top5classes'])
#df3 = pd.read_csv('/data/omran/cities_data/dataset/Mapillary.csv' ,usecols=['IMG_ID','Top5classes'])

#database = pd.concat([df1, df2, df3])
#database = database.drop_duplicates(subset=['IMG_ID'])

#database['Top1'] = ''
#database['Top2'] = ''
#database['Top3'] = ''
#database['Top4'] = '' 
#database['Top5'] = '' 



#def func(database):

#    for index, row in tqdm(database.iterrows(),  total=(database.shape[0])):
#        Top5 = (row.Top5classes[1:])[:-1].split(',')
#        database.at[index,'Top1']= int(Top5[0])
#        database.at[index,'Top2']= int(Top5[1])
#        database.at[index,'Top3']= int(Top5[2])
#        database.at[index,'Top4']= int(Top5[3])
#        database.at[index,'Top5']= int(Top5[4])

#    return database    



#def parallelize_dataframe(df, func, n_cores=24):  # os.cpu_count()-4

#    df_split = np.array_split(df, n_cores)
#    pool = Pool(n_cores)
#    df = pd.concat(pool.map(func, df_split))
#    pool.close()
#    pool.join()
#    return df


#database = parallelize_dataframe(database, func)




#database.to_csv('/data/omran/cities_data/dataset/Database_Top5.csv', index=False)
database = pd.read_csv('/data/omran/cities_data/dataset/Database_Top5.csv' , low_memory=False, usecols=['IMG_ID','Top1','Top2','Top3','Top4','Top5'])

database['IMG_ID'] = database['IMG_ID'].str.replace('-','_',1)

#database['IMG_ID'].astype('str')

print(database.dtypes)
print(database.head())

images_folder    = '/data/omran/cities_data/dataset/cities/'
classes_database = '/data/omran/siamese_cities/dataset_meta/cities_csv/'

output_folder    = '/data/omran/cities_data/dataset/cities/csv_meta'
  
#city_data = pd.read_csv(os.path.join(classes_database, city),usecols=['IMG_ID','Top5classes']) 
#city_data.drop_duplicates(subset=['IMG_ID'],inplace=True)

def non_match_elements(list_a, list_b):
    non_match = []
    for i in list_a:
        if i not in list_b:
            non_match.append(i)
    return non_match

for city in (os.listdir(classes_database)):
#for city in ['Cairo.csv']:
    city_name = city[:-4]

    for set in ['test','training','validation']:

        print(city_name)

        set_folder = os.path.join(images_folder , set , city_name)

        #print(set_folder)

        set_list = [str(f[:-5])  for f in os.listdir(set_folder) if os.path.isfile(os.path.join(set_folder, f))]
        print("set_list",len(set_list))

        #print(set_list)
        #print(city_data.head())
        
        #city_data_set = database.query('IMG_ID in @set_list')

        city_data_set = database[database['IMG_ID'].isin(set_list)]

        image_top5_list = city_data_set.IMG_ID.values.tolist()
        print(len(image_top5_list))
        
        print(city_data_set.shape[0])
        
        print(non_match_elements(set_list,image_top5_list))

        #city_data_set = city_data_set.drop_duplicates(subset=['IMG_ID'])

        city_data_set.to_csv(str(os.path.join(output_folder , set )) + '/' +  (city), index=False)

        #break




