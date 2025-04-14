from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from collections import defaultdict
import logging

from tqdm import tqdm

from .adt import Validator, Model
from transferwareai.config import settings


class GenericValidator(Validator):
    """Applies the validation strategy entirely on the model interface."""

    def __init__(self, device: str):
        super().__init__()
        self.device = device

    def validate(
        self, model: Model, validation_set: ImageFolder
    ) -> tuple[dict[int, float], float]:
        logging.debug("Starting generic validation")
        num_correct = 0
        class_val = defaultdict(list)

        dl = DataLoader(validation_set, shuffle=False, batch_size=1, num_workers=5)
        # Validation folder is pattern ID as class, so create mapping where we can get the
        # Pattern ID of the image in the validation loop
        idx_to_class = {j: i for (i, j) in validation_set.class_to_idx.items()}

        for img, id in tqdm(dl):
            img = img.to(self.device)

            matches = model.query(img[0], top_k=settings.query.top_k + 20)

            clean_matches = []
            ids = []
            for image in matches:
                if len(clean_matches) < settings.query.top_k:
                    if image.id not in ids:
                        clean_matches.append(image)
                        ids.append(image.id)
                if len(clean_matches) >= settings.query.top_k:
                    break

            matches = clean_matches

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
