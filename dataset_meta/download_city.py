""" Download images per city """

from argparse import ArgumentParser
from io import BytesIO
from pathlib import Path
import time
from multiprocessing import Pool
from functools import partial
import logging
import requests
import pandas as pd
import PIL
from PIL import ImageFile
from tqdm import tqdm
from datetime import datetime
from sklearn.utils import shuffle
import pandas as pd
import requests
import time
import numpy as np
import os
from sklearn.utils import shuffle
import sys
import fnmatch
ImageFile.LOAD_TRUNCATED_IMAGES = True

def image_download(x, size_suffix, min_edge_size, access_token):

    try:

        url_original = x["url"]

        if "flickr" in url_original:

            #logger.info("Download flickr")
            if "live" in url_original:
                return flickr_download(x, size_suffix="", min_edge_size=min_edge_size)
            else:
                return flickr_download(x, size_suffix=size_suffix, min_edge_size=min_edge_size)
            # return None
        elif "unsplash" in url_original:
            #logger.info("Download unsplash")
            return unsplash_download(x, min_edge_size)
            # return None
        elif "mapillary" in url_original:
            return download_mapillary(x, min_edge_size, access_token)
        else:
            logger.error(f"Invalid URL : {url_original}")
            return None
    except:
        logger.error(f"Could not dowonload Image : {url_original}")
        return None


def _thumbnail(img: PIL.Image, size: int) -> PIL.Image:
    # resize an image maintaining the aspect ratio
    # the smaller edge of the image will be matched to 'size'
    PIL.Image.MAX_IMAGE_PIXELS = None
    size = args.size
    w, h = img.size
    if (w <= size) or (h <= size):
        return img
    if w < h:
        ow = size
        oh = int(size * h / w)
        return img.resize((ow, oh), PIL.Image.BILINEAR)
    else:
        oh = size
        ow = int(size * w / h)
        return img.resize((ow, oh), PIL.Image.BILINEAR)


def download_mapillary(x, min_edge_size, access_token):

    image_id = x["image_id"]
    url = x["url"]

    header = {'Authorization': 'OAuth {}'.format(access_token)}

    try:
        r = requests.get(url, headers=header, timeout=30)
        data = r.json()
        image_url = data['thumb_1024_url']
        # save each image with ID as filename to directory by sequence ID
        # time.sleep(0.01)
        r = requests.get(image_url, timeout=30)

    except:
        time.sleep(60)
        logger.warning(
            f"To many requests, sleep for 60s... {image_id} : {url}")
        try:
            r = requests.get(url, headers=header, timeout=30)
            data = r.json()
            image_url = data['thumb_1024_url']
            # save each image with ID as filename to directory by sequence ID
            # time.sleep(1)
            r = requests.get(image_url, timeout=30)
        except:
            logger.error(f'Can not download iamge {image_url}')
            return None

    if r:
        try:
            image = PIL.Image.open(BytesIO(r.content))
        except PIL.UnidentifiedImageError as e:
            logger.error(f"{image_id} : {url}: {e}")
            return None
    else:
        logger.error(f"{image_id} : {url}: {r.status_code}")
        return None

    if image.mode != "RGB":
        image = image.convert("RGB")

    try:
        image = _thumbnail(image, min_edge_size)
    except:
        logger.warning(f"Image could not be resized {image_id} : {url}")
        return None

    # convert to jpeg
    fp = BytesIO()
    image.save(fp, "JPEG")

    #   image.save(fp, "JPEG")
    #    image.save(output_file + '{}.jpeg'.format(str(image_id).replace("/", "_")))
    # image.save(
    #    '/data2/omran/models/delete/{}.jpeg'.format(image_id))
    #raw_bytes = fp.getvalue()
    return {"image": image, "id": image_id}


def unsplash_download(x, min_edge_size):

    # prevent downloading in full resolution using size_suffix
    # https://www.flickr.com/services/api/misc.urls.html

    image_id = x["image_id"]
    url = x["url"]
    # if size_suffix != "":
    #    url = url_original
    # modify url to download image with specific size
    #    ext = Path(url).suffix
    #    url = f"{url.split(ext)[0]}_{size_suffix}{ext}"
    # else:
    #    url = url_original

    try:
        r = requests.get(url, timeout=30)
    except requests.exceptions.ConnectionError:
        time.sleep(60)
        logger.warning(
            f"To many requests, sleep for 60s... {image_id} : {url}")
        try:
            r = requests.get(url, timeout=30)
        except:
            return None

    if r:
        try:
            image = PIL.Image.open(BytesIO(r.content))
        except PIL.UnidentifiedImageError as e:
            logger.error(f"{image_id} : {url}: {e}")
            return
    elif r.status_code == 500 or r.status_code == 503:
        time.sleep(60)
        logger.warning("To many requests, sleep for 60s...")
        return unsplash_download(x, min_edge_size)
    else:
        #logger.error(f"{image_id} : {url}: {r.status_code}")
        return None

    if image.mode != "RGB":
        image = image.convert("RGB")

    # resize if necessary
    try:
        image = _thumbnail(image, min_edge_size)
    except:
        logger.warning(f"Image could not be resized {image_id} : {url}")
        return None

    # convert to jpeg
    fp = BytesIO()
    image.save(fp, "JPEG")
    #logger.info("Save unsplash image")
    # image.save(
    #    '/media/hdddati1/omran/GeoEstimation/resources/images/unsplash/{}.jpeg'.format(image_id))
    #raw_bytes = fp.getvalue()
    return {"image": image, "id": image_id}


def flickr_download(x, size_suffix, min_edge_size):

    # prevent downloading in full resolution using size_suffix
    # https://www.flickr.com/services/api/misc.urls.html

    image_id = x["image_id"]
    url_original = x["url"]
    if size_suffix != "":
        url = url_original
        # modify url to download image with specific size
        ext = Path(url).suffix
        url = f"{url.split(ext)[0]}_{size_suffix}{ext}"
    else:
        url = url_original

    try:
        r = requests.get(url, timeout=30)
    except requests.exceptions.ConnectionError:
        time.sleep(60)
        logger.warning(
            f"To many requests, sleep for 60s... {image_id} : {url}")
        try:
            r = requests.get(url, timeout=30)
        except:
            return None

    if r:
        try:
            image = PIL.Image.open(BytesIO(r.content))
        except PIL.UnidentifiedImageError as e:
            logger.error(f"{image_id} : {url}: {e}")
            return None
    elif r.status_code == 129:
        time.sleep(60)
        logger.warning("To many requests, sleep for 60s...")
        return flickr_download(x, min_edge_size, size_suffix)
    else:
        #logger.error(f"{image_id} : {url}: {r.status_code}")
        return None

    if image.mode != "RGB":
        image = image.convert("RGB")

    # resize if necessary
    image = _thumbnail(image, min_edge_size)
    # convert to jpeg
    fp = BytesIO()
    image.save(fp, "JPEG")
    # image.save(
    #    '/media/hdddati1/omran/GeoEstimation/resources/images/flicker/{}.jpeg'.format(str(image_id).replace("/", "_")))
    #raw_bytes = fp.getvalue()
    return {"image": image, "id": image_id}


def urls_one_city(lat, lng, dist_out, im_splash, im_flickr, im_mapillary, url_Flickr, url_Splash, url_Mapillary):

    Flickr_data    = pd.read_csv(url_Flickr)
    Splash_data    = pd.read_csv(url_Splash)
    Mapillary_data = pd.read_csv(url_Mapillary)

    # Get number of images withen the bounding box of a given city
    west, south, east, north = [lng-dist_out, lat -
                                dist_out, lng+dist_out, lat+dist_out]

    city_Flickr = Flickr_data.query(
        'LON > @west and LON < @east and LAT > @south and LAT < @north')
    city_Splash = Splash_data.query(
        'LON > @west and LON < @east and LAT > @south and LAT < @north')
    city_Mapillary = Mapillary_data.query(
        'LON > @west and LON < @east and LAT > @south and LAT < @north')

    city_Splash = shuffle(city_Splash)
    city_Flickr = shuffle(city_Flickr)
    city_Mapillary = shuffle(city_Mapillary)

    if(im_splash > city_Splash.shape[0]):
        im_splash = city_Splash.shape[0]

    if(im_flickr > city_Flickr.shape[0]):
        im_flickr = city_Flickr.shape[0]   

    if(im_mapillary > city_Mapillary.shape[0]):
        im_mapillary = city_Mapillary.shape[0]

    city_Splash = city_Splash.sample(im_splash, replace=True)
    city_Flickr = city_Flickr.sample(im_flickr, replace=True)
    city_Mapillary = city_Mapillary.sample(im_mapillary, replace=True)

    out_df = pd.concat([city_Splash, city_Flickr, city_Mapillary])

    city_df = shuffle(out_df)

    return (city_df)


class ImageDataloader:
    def __init__(self, url_df, shuffle=False, nrows=None):

        logger.info("Read dataset")

        self.df =  url_df[["IMG_ID", "url"]].copy()
        self.df.rename(columns={'IMG_ID': 'image_id'}, inplace=True)
        # remove rows without url
        self.df = self.df.dropna()
        if shuffle:
            #logger.info("Shuffle images")
            self.df = self.df.sample(frac=1, random_state=10)
        #logger.info(f"Number of URLs: {len(self.df.index)}")

    def __len__(self):
        return len(self.df.index)

    def __iter__(self):
        for image_id, url in zip(self.df["image_id"].values, self.df["url"].values):
            yield {"image_id": image_id, "url": url}


def parse_args():

    args = ArgumentParser()
    args.add_argument(
        "--threads",
        type=int,
        default=24,
        help="Number of threads to download and process images",
    )
    args.add_argument(
        "--output",
        type=Path,
        default=Path(
            "/data/omran/cities_data/dataset/open_set"),
        help="Output directory where images are stored",
    )
    args.add_argument(
        "--input_cities",
        type=Path,
        default=Path(
            "/data/omran/siamese_cities/dataset_meta/cities.csv"),
    )

    args.add_argument(
        "--url_Flickr",
        type=Path,
        default=Path("/data/omran/cities_data/dataset/Flickr.csv"),
    )
    args.add_argument(
        "--url_Splash",
        type=Path,

        default=Path("/data/omran/cities_data/dataset/Splash.csv"),
    )
    args.add_argument(
        "--url_Mapillary",
        type=Path,

        default=Path("/data/omran/cities_data/dataset/Mapillary.csv"),
    )
    args.add_argument(
        "--size",
        type=int,
        default=640,
        help="Rescale image to a minimum edge size of SIZE",
    )
    args.add_argument(
        "--max_img",
        type=int,
        default=150000,
        help="Maximum number of images per city",
    )
    args.add_argument(
        "--size_suffix",
        type=str,
        default="z",
        help="Image size suffix according to the Flickr API; Empty string for original image",
    )
    args.add_argument(
        "--access_token",
        type=str,
        default="MLY|5110030895754170|06d86ff4d13808be555454636bd064c1",
        help="Access token for Mapillary API",
    )
    args.add_argument("--nrows", type=int)
    args.add_argument(
        "--shuffle", action="store_false", help="Shuffle list of URLs before downloading"
    )
    return args.parse_args()


def main():

    # Read cities dataset get the GPS, box dim, and number of images per set
    df_input_cities = pd.read_csv(args.input_cities)

    for index, row in df_input_cities.iterrows():

        logger.info(f"Downloading images per {row.city}")

        output_folder = str(args.output) + f'/{row.city}/'

        if not os.path.exists(output_folder):
            # Create a new directory because it does not exist
            os.makedirs(output_folder)

        # get one data frame pre city
        df_one_city = urls_one_city(row.lat, row.lng, row.dist_out , row.im_splash  , row.im_flickr + 300,
                                    row.im_mapillary  + 300 , args.url_Flickr , args.url_Splash, args.url_Mapillary)

        # save the dataframe per city in this location
        df_one_city.to_csv(
            f'/data/omran/cities_data/dataset/cities_csv/{row.city}.csv', index=False)

        image_loader = ImageDataloader(
            df_one_city, nrows=args.nrows, shuffle=args.shuffle)
        counter_successful = 0
        with Pool(args.threads) as p:
            start = time.time()

            for i, x in enumerate(
                p.imap(
                    partial(
                        image_download,
                        size_suffix=args.size_suffix,
                        min_edge_size=args.size,
                        access_token=args.access_token,
                    ),
                    image_loader,
                )
            ):
                if x is None:
                    continue
                #print('i',i)
                #print('x',x)
                x["image"].save(output_folder + '{}.jpeg'.format(str(x["id"]).replace("/", "_")))        
                #f.write(x)
                counter_successful += 1
                if (len(fnmatch.filter(os.listdir(output_folder), '*.jpeg')) >= args.max_img ):
                    break
                # print(i)
                #if i % 10000 == 0:
                #    end = time.time()
                #    logger.info(
                #        f"Sucesfully downloaded {counter_successful}/{len(image_loader)} ")
                #    logger.info(f"{i}: {1000 / (end - start):.2f} image/s")
                #    start = end
        logger.info(
            f"{row.city} : Sucesfully downloaded {counter_successful}/{len(image_loader)} images ({counter_successful / len(image_loader):.3f})"
    )
    return 0

if __name__ == "__main__":

    args = parse_args()
    logger = logging.getLogger("ImageDownloader")
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler("writer.log")
    fh.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)

    sys.exit(main())
