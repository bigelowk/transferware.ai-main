from pathlib import Path
import logging
import warnings

from transferwareai.config import settings
from transferwareai.models.construct import get_abstract_factory
from torchvision.datasets import ImageFolder

import torchvision.io

from transferwareai.models.adt import Model
from collections import defaultdict
from torch.utils.data import DataLoader
from tqdm import tqdm

from PIL.Image import Image
from transferwareai.models.adt import Model, Trainer, AbstractModelFactory, ImageMatch
from torch import Tensor
import torch

from transferwareai.models.zhaomodel import ZhaoModel

# This script is to validate the current model without having to run the entire training script

# Set loggers to not spam
warnings.filterwarnings("ignore", ".*(Tensor|tensor).*")
logging.getLogger("PIL").setLevel(logging.WARN)

logging.getLogger().setLevel(logging.DEBUG)


def scrap():
    # query the validation images from the folder
    def validate(model: Model, validation_set: ImageFolder    ) -> tuple[dict[int, float], float]:
        logging.debug("Starting generic validation")
        num_correct = 0
        class_val = defaultdict(list)

        dl = DataLoader(validation_set, shuffle=False, batch_size=1, num_workers=5)
        # Validation folder is pattern ID as class, so create mapping where we can get the
        # Pattern ID of the image in the validation loop
        idx_to_class = {j: i for (i, j) in validation_set.class_to_idx.items()}

        for img, id in tqdm(dl):
            img = img.to("cpu")

            matches = model.query(img[0])
            id = int(idx_to_class[int(id[0])])

            # Check if correct id is in top 10 matches
            found = False
            for match in matches:
                if match.id == id:
                    num_correct += 1
                    found = True
                    break

            class_val[id].append(int(found))

        # Calc per class accuracy
        class_percents = {}
        for id, finds in class_val.items():
            tp = sum(finds)
            perc = tp / len(finds)
            class_percents[id] = perc

        return class_percents, num_correct / len(validation_set)

    # get the factory for the model implementation
    factory = get_abstract_factory(settings.model_implimentation, "query")

    # create a model class with the current model information
    model = factory.get_model()

    # define the resource directory for the validation images
    res_path = Path(settings.training.validation_dir)

    # create the validation set
    valid_ds = ImageFolder(str(res_path.absolute()))

    # create a validator class for the current model
    validator = factory.get_validator()

    # call the validator
    class_val, val_percent = validator.validate(model, valid_ds)

    print(f"Validator class: {class_val}")
    print(f"Validator class: {val_percent}")

    # class_val, val_percent = validate(model, valid_ds)
    #
    # print(f"Validator function: {class_val}")
    # print(f"Validator function: {val_percent}")


    # query an individual image
    t = torchvision.io.read_image(
        "../scratch/val_hold/56311/8055.jpg", torchvision.io.ImageReadMode.RGB
    )

    # Normal amount
    res = model.query(t, top_k=10)

    pass






if __name__ == "__main__":

    def query(self, image: Tensor | Image, top_k: int = 10) -> list[ImageMatch]:
        with torch.no_grad():
            # Preprocess
            # image = self.model.transform(image)
            image = image.to(self.device).float()

            embedding = self.model.get_embedding(image)

            nns, dists = self.index.get_nns_by_vector(
                embedding.cpu().detach(), top_k, include_distances=True
            )
            matches = [
                ImageMatch(id=self.annoy_id_to_pattern_id[nn], confidence=dist)
                for nn, dist in zip(nns, dists)
            ]

            return matches


    # do this to avoid transforming the images twice
    ZhaoModel.query = query


    # get the factory for the model implementation
    factory = get_abstract_factory(settings.model_implimentation, "query")

    # create a model class with the current model information
    model = factory.get_model()

    # define the resource directory for the validation images
    res_path = Path(settings.training.validation_dir)

    # create the validation set
    valid_ds = ImageFolder(str(res_path.absolute()))

    # create a validator class for the current model
    validator = factory.get_validator()

    # call the validator
    class_val, val_percent = validator.validate(model, valid_ds)

    print(f"Validator class: {class_val}")
    print(f"Validator class: {val_percent}")

