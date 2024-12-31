from abc import ABC, abstractmethod
from pathlib import Path

from PIL.Image import Image
from pydantic import BaseModel
from torch import Tensor
from dataclasses import dataclass

from torchvision.datasets import ImageFolder

from ..data.dataset import CacheDataset


class ImageMatch(BaseModel):
    """An image matching to the query image/"""

    id: int
    """Id of matching image"""
    confidence: float
    """Confidence metric of matching image. Could be 0-1, or a distance, depending on model."""


class Model(ABC):
    """Interface for query models."""

    @abstractmethod
    def query(self, image: Tensor | Image, top_k: int = 10) -> list[ImageMatch]:
        """Takes the input image, and finds the k closest images from the dataset used for training."""
        ...

    @abstractmethod
    def get_resource_files(self) -> list[Path]:
        """
        Returns paths to all resource files associated with this model. Can be used to send to the query api after
        training is complete.
        """
        ...


class Trainer(ABC):
    """Interface for methods of training models."""

    @abstractmethod
    def train(self, dataset: CacheDataset) -> Model:
        """Creates a model, training it and creating related files like caches."""
        ...


class Validator(ABC):
    """Interface for validation techniques."""

    @abstractmethod
    def validate(
        self, model: Model, validation_set: ImageFolder
    ) -> tuple[dict[int, float], float]:
        """
        Validates the model against real world, untrained data. Returns validation percent, and a dict of accuracy
        per id. This is a validation of the model in the final system,
        rather than validation in a deep learning context.
        """
        ...


class AbstractModelFactory(ABC):
    """Interface for creating families of query models."""

    def __init__(self, resource_path: Path) -> None:
        """
        :param resource_path: Path to folder where all assets required for training are stored, and where model files
        will be placed. Required files will change depending on the implementation used.
        """
        self._resource_path = resource_path

    @abstractmethod
    def get_model(self) -> Model: ...

    @abstractmethod
    def get_trainer(self) -> Trainer: ...

    @abstractmethod
    def get_validator(self) -> Validator: ...
