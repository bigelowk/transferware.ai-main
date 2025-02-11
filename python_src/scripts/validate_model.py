"""The purpose of this script is to be able to run the validation set on the current model
without having to enter a training job first"""

from pathlib import Path
import logging
import warnings

from transferwareai.config import settings
from transferwareai.models.construct import get_abstract_factory

from torchvision.datasets import ImageFolder
from torchvision import transforms

from transferwareai.models.zhaomodel import ZhaoModel
from transferwareai.models.zhaomodel import EmbeddingsValidator
from transferwareai.models.generic import GenericValidator


# Set loggers to not spam
warnings.filterwarnings("ignore", ".*(Tensor|tensor).*")
logging.getLogger("PIL").setLevel(logging.WARN)

logging.getLogger().setLevel(logging.DEBUG)



if __name__ == "__main__":

    def validate(
            self, model: ZhaoModel, validation_set: ImageFolder
    ) -> tuple[dict[int, float], float]:

        transform = transforms.Compose([transforms.ToTensor()])
        validation_set.transform = transform

        return GenericValidator(self.device).validate(model, validation_set)

    EmbeddingsValidator.validate = validate


    # get the factory for the model implementation
    factory = get_abstract_factory(settings.model_implimentation, "training")

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

