from torch.utils.data import Subset, Dataset
from sklearn.model_selection import train_test_split
import polars as pl

from transferwareai.data.dataset import CacheDataset
from transferwareai.tccapi.api_cache import ApiCache


def create_test_train_split(
    data: CacheDataset, test_size: float
) -> tuple[Dataset, Dataset]:
    """Creates a stratified test and train split of the dataset."""

    # generate indices: instead of the actual data we pass in integers instead
    train_indices, test_indices, _, _ = train_test_split(
        range(len(data)),
        data.targets,
        stratify=data.targets,
        test_size=test_size,
    )

    # generate subset based on indices
    train_split = Subset(data, train_indices)
    test_split = Subset(data, test_indices)

    return test_split, train_split


def pattern_number_to_id(api: ApiCache, pattern_id: int) -> int:
    """Converts a pattern id to its corresponding id in the DB. Pattern ids are what are found on the TCC website,
    the other id is the one used in this tool."""
    df = api.as_df()
    res = (
        df.select(pl.col("pattern_number"), pl.col("id")).filter(
            pl.col("pattern_number") == pattern_id
        )
    )["id"][0]
    return int(res)
