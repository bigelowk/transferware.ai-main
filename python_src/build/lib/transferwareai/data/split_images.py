"""The purpose of this file is to split pattern images into smaller images
for the purposes of data augmentation"""

import cv2

from pathlib import Path
import polars as pl
import logging

from transferwareai.config import settings
from transferwareai.data.dataset import CacheDataset
from transferwareai.tccapi.api_cache import ApiCache



def split(ds, specs: list = []):
    """Splits an image corresponding to the image tag into the rows and columns. Does this for all the images in the
    given api dataset"""

    api = ds._cache
    # for all the images in the api, split them into pieces
    for id in api.as_df()["id"]:
        for i in range(len(specs)):
            tag = specs[i][0]
            img_path = api.get_image_file_path_for_tag(id, tag)
            img = cv2.imread(str(img_path))

            # check if an image with the given tag exists, if not then don't execute this code chunk
            if img is not None:
                paths = []
                cat = ds._class_labels.select(pl.col("category").filter(pl.col("id") == id))["category"][0]


                img_folder = api.get_image_path_for_id(id)
                height, width, channels = img.shape

                # image splitting code below is adapted from: https://www.tutorialspoint.com/dividing-images-into-equal-parts-using-opencv-python
                H_SIZE = specs[i][1]
                W_SIZE = specs[i][2]

                for row in range(H_SIZE):
                    for col in range(W_SIZE):
                        x = width / W_SIZE * col
                        y = height / H_SIZE * row
                        h = (height / H_SIZE)
                        w = (width / W_SIZE)

                        temp_img = img[int(y):int(y + h), int(x):int(x + w)]

                        # write the image to the folder if one does not already exist
                        # to keep these folders from getting too messy, this splitting should only happen once
                        # if it is to be repeated, delete the asset files and redownload them to set them back to default
                        temp_img_path = img_folder.joinpath(f"{id}-{tag}-row{row}-col{col}.jpg")
                        paths.append(str(temp_img_path))

                        cv2.imwrite(temp_img_path, temp_img)

                temp_ds = pl.DataFrame({"id": [id]*(H_SIZE*W_SIZE), "image_url": paths, "category": [cat]*(H_SIZE*W_SIZE)})
                ds.append_paths(temp_ds)

if __name__ == '__main__':

    res_path = Path("../../../scratch")
    # Update TCC cache
    api = ApiCache.from_cache(res_path.joinpath("cache"), no_update=True)

    subset = True
    if subset:
        logging.info("Subsetting dataset")
        val_ids = [47293]
        api.subset(10, val_ids)

    ds = CacheDataset(api, skip_ids=[67001])

    split(ds, [['pattern', 2, 2]])



