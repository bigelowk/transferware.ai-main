import logging
from typing import Optional

import torch
from torch.utils.data import Dataset
from torch import Tensor

from transferwareai.tccapi.api_cache import ApiCache
import polars as pl
from functools import lru_cache
import PIL.Image


class CacheDataset(Dataset):
    """Dataset wrapping the TCC api cache."""

    def __init__(
        self, cache: ApiCache, transform=None, skip_ids: Optional[list[int]] = None
    ) -> None:
        """
        Create a Dataset wrapping the TCC api.
        :param cache: TCC api cache.
        :param transform: Transforms to apply to each image of the dataset, as they are loaded.
        :param skip_ids: Ids to drop from the dataset.
        """

        self._cache = cache
        self._transform = transform
        self._df = cache.as_df()

        # ID to category label, ordered by IDs ascending
        self._class_labels = (
            self._df.select(
                pl.col("id"),
                pl.col("category")
                .list.eval(pl.element().struct.field("name"))
                .list.first(),
            )
            .drop_nulls()
            .filter(~pl.col("id").is_in(skip_ids or []))
            .sort(by=pl.col("id"))
        )

        def map_to_paths(row: tuple):
            id, tag = row
            return id, str(self._cache.get_image_file_path_for_tag(id, tag).absolute())

        # IDs to category and each image file per id
        self._image_paths = (
            (
                self._df.select(
                    pl.col("id"),
                    pl.col("images")
                    .list.eval(pl.element().struct.field("tags"))
                    .alias("tags"),
                )
                .drop_nulls()
                .explode("tags")
                .map_rows(map_to_paths)
            )
            .rename({"column_0": "id", "column_1": "image_url"})
            .filter(~pl.col("id").is_in(skip_ids or []))
            .join(self._class_labels, on=pl.col("id"))
            .sort(by="id")
        )

        # class to ID (ID to class is just the list)
        self._class_ids = {cat: i for i, cat in enumerate(self.class_labels())}

    @lru_cache(maxsize=1)
    def class_labels(self) -> list[str]:
        """Gets the class labels for the dataset."""
        return self._class_labels["category"].unique().sort().to_list()

    def class_num(self) -> int:
        """Gets the number of classes for the dataset."""
        return len(self.class_labels())

    def class_id_for_category(self, category: str) -> int:
        """Gets the class id for category."""
        return self._class_ids[category]

    def category_for_id(self, id: int) -> str:
        """Gets category for a given class id."""
        return self.class_labels()[id]

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        """Gets the nth sample of (image, category id)"""
        _id, path, cat = self._image_paths[idx]

        # Explicitly load image as RGB, else we get alpha channels
        im = PIL.Image.open(path[0]).convert("RGB")

        if self._transform:
            im = self._transform(im)

        id_tensor = torch.tensor(self.class_id_for_category(cat[0]), dtype=torch.long)

        return im, id_tensor

    def __len__(self) -> int:
        """Total number of images in the dataset."""
        return len(self._image_paths)

    @property
    @lru_cache(maxsize=1)
    def targets(self) -> list[int]:
        """Returns the class id for each sample in the dataset."""
        return [self.class_id_for_category(c) for c in self._image_paths["category"]]

    def set_transforms(self, transforms):
        """Sets the transformation to be applied to all images in dataset."""
        self._transform = transforms

    @lru_cache(maxsize=1)
    def get_pattern_ids(self) -> list[int]:
        """
        Returns the ids of patterns used in samples, in the order of get index. This means ids will duplicate for as
        many images there are for a pattern.
        """
        return self._image_paths["id"].to_list()
