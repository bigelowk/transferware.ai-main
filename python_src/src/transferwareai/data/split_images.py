"""The purpose of this file is to split pattern images into smaller images
for the purposes of data augmentation"""

import cv2

from pathlib import Path

from transferwareai.config import settings
from transferwareai.tccapi.api_cache import ApiCache
import logging


# TODO - where to implement this in the script? --> Training Script between the image subset and the cache dataset


def split_image(api, h_split = 2, v_split = 2, tag = 'pattern'):
    """Splits the image, given the pattern id and image tag, into h_split rows and v_split columns"""


    # Number of pieces Horizontally
    W_SIZE = h_split
    # Number of pieces Vertically to each Horizontal
    H_SIZE = v_split

    # for all the images in the api, split them into pieces
    for id in api.as_df()["id"]:
        img_path = api.get_image_file_path_for_tag(id, tag)
        img = cv2.imread(str(img_path))

        # if an image with the given tag does not exist, skip it
        if img is not None:
            img_folder = api.get_image_path_for_id(id)
            height, width, channels = img.shape

            # image splitting code below is adapted from: https://www.tutorialspoint.com/dividing-images-into-equal-parts-using-opencv-python
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
                    cv2.imwrite(temp_img_path, temp_img)

if __name__ == '__main__':

    res_path = Path("../../../scratch")
    # Update TCC cache
    api = ApiCache.from_cache(res_path.joinpath("cache"), no_update=True)

    subset = True
    if subset:
        logging.info("Subsetting dataset")
        val_ids = [47293]
        api.subset(0, val_ids)

    split_image(api, 2, 2, 'pattern')


