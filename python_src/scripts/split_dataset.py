from pathlib import Path
import logging
import torch
import polars as pl

from transferwareai.config import settings
from transferwareai.data.dataset import CacheDataset
from transferwareai.tccapi.api_cache import ApiCache


res_path = Path(settings.training.resource_dir)
api = ApiCache.from_cache(res_path.joinpath("cache"), no_update=True)

# TODO: add to api_cache
def _set_df(self, df):
    self._df = df

# TODO: add to api_cache
def subset(self, n, val_ids):
    keep = self._df.filter(pl.col("id").is_in(val_ids or []))
    df = self._df.sample(n, seed=314)
    new = pl.concat([df, keep]).unique()
    self._set_df(new)

ApiCache.subset = subset
ApiCache._set_df = _set_df

# TODO: add to training_script
if settings.training.subset:
    logging.info("Subsetting dataset")
    api.subset(settings.training.subset_n, settings.training.val_ids)

# check to see that it was subset
print(len(api._df))

ds = CacheDataset(api, skip_ids=settings.training.skip_ids)

# check to see if changes were propagated to cache dataset
print(ds._class_labels.head())
print(ds._image_paths.head())
print(len(ds))


# subset the data set
# split the images using pattern image
# name each image with a number as a tag
# append the image urls to the image_paths
# return the df

