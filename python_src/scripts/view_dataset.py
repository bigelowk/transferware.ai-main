from transferwareai.config import settings
from pathlib import Path
from transferwareai.data.dataset import CacheDataset
from transferwareai.tccapi.api_cache import ApiCache
from transferwareai.data.split_images import split


"""This script is to help get a better idea of what the Api and Cache datasets look like """

res_path = Path(settings.query.resource_dir)
# Get cache
api = ApiCache.from_cache(res_path.joinpath("cache"), no_update=True)
ds = CacheDataset(api, skip_ids=settings.training.skip_ids)

#split(ds, [['pattern', 2, 2]])

print(f"Dataframe Shape: {ds._df.shape}")
print("Dataframe:")
print(ds._df.head(20))

print(f"Class Labels Shape: {ds._class_labels.shape}")
print("Class Labels:")
print(ds._class_labels.head())

print(f"Image Paths Shape: {ds._image_paths.shape[0]}")
print("Image Paths:")
print(ds._image_paths.head())

print(f"Class Ids Length: {len(ds._class_ids)}")
print("Class Ids:")
print(ds._class_ids)
