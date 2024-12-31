# This needs to be its own file, as otherwise we have circular imports
from pathlib import Path
from typing import Literal

from transferwareai.models.adt import AbstractModelFactory
from transferwareai.models.zhaomodel import ZhaoModelFactory
from transferwareai.config import settings


def get_abstract_factory(
    name: str, mode: Literal["query", "training"]
) -> AbstractModelFactory:
    """
    Returns a factory matching the string passed. Name is the name of the abstract factory implementation class.
    Mode is the current mode of the system, which will change the settings passed to the factory.
    """
    sett = settings[mode]

    match name:
        case "ZhaoModelFactory":
            return ZhaoModelFactory(Path(sett.resource_dir), sett.torch_device)
