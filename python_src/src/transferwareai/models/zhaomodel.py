# This file adapts the method by Zhao et. al. in https://doi.org/10.1016/j.daach.2023.e00269
import logging
from abc import ABCMeta, abstractmethod
from pathlib import Path
from typing import Optional, Callable, Type, TypeVar

from PIL.Image import Image
import annoy
import torch
import torchvision
from torch import Tensor
from torch.utils.data import DataLoader, RandomSampler
from torchvision.datasets import ImageFolder
from torchvision.transforms import v2 as transforms
from torchvision.models.feature_extraction import create_feature_extractor
from torchmetrics.classification import BinaryAveragePrecision, BinaryAccuracy
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from .adt import Model, Trainer, AbstractModelFactory, ImageMatch
from .generic import GenericValidator
from .utils import create_test_train_split
from ..data.dataset import CacheDataset
from ..config import settings


class EmbeddingsModelImplementation(metaclass=ABCMeta):
    """Superclass for implementations of torch models for getting embeddings."""

    def __init__(self):
        self._augmentations = None
        self._training = False

    @abstractmethod
    def transform(self, img: Tensor | Image):
        """Preprocesses input, applying augmentations if in training mode."""
        pass

    def training_mode(self):
        """Toggles augmentations on."""
        self._training = True

    def eval_mode(self):
        """Toggles augmentations off."""
        self._training = False

    def add_augmentations(self, augmentations: transforms.Transform):
        """
        Adds augmentation function.
        This function will be called during transform calls only when in training mode.
        """
        self._augmentations = augmentations

    @abstractmethod
    def get_embedding(self, image: Tensor | Image):
        """Gets the embedding vector for the given image in (C, W, H). The image is assumed to be preprocessed."""
        pass

    @abstractmethod
    def embedding_size(self) -> int:
        """Size of the embedding vector."""
        ...


class ZhaoVGGModel(EmbeddingsModelImplementation):
    """Original VGG model from the Zhao paper."""

    def embedding_size(self) -> int:
        return 4096

    def __init__(
        self,
        class_count: int,
        weights: Optional[Path] = None,
        pretrained: bool = False,
        device="cpu",
    ):
        """
        Wraps the torch model.
        :param class_count: Number of classes
        :param weights: Path to weights file to load
        :param pretrained: if weights are trained for transferware, or are general pretrained weights
        :param device: Device to load on
        """
        super().__init__()

        self.device = device
        # Use torch weights if we are using a pretrained weights and passed none
        model_vgg16 = torchvision.models.vgg16_bn(
            weights=(
                torchvision.models.VGG16_BN_Weights.IMAGENET1K_V1
                if pretrained and weights is None
                else None
            )
        )

        # Loading our trained model, so need to change shape
        if not pretrained:
            model_vgg16.classifier[6] = torch.nn.Linear(4096, class_count)

        # Load trained model if provided
        if weights:
            model_vgg16_pth = torch.load(weights)
            model_vgg16.load_state_dict(model_vgg16_pth)

        if pretrained:
            model_vgg16.classifier[6] = torch.nn.Linear(4096, class_count)

        self.model = model_vgg16.to(device)

        # Preprocessing steps
        self._transform = transforms.Compose(
            [
                transforms.Grayscale(3),
                transforms.Resize((224, 224)),
                torchvision.models.VGG16_BN_Weights.IMAGENET1K_V1.transforms().to(
                    device
                ),
            ]
        )

    def transform(self, img: Tensor | Image) -> Tensor:
        """Preprocesses input, applying augmentations if in training mode."""
        # First convert to tensor
        match img:
            case Image():
                temp = transforms.ToTensor()(img)
            case Tensor():
                temp = img
            case _:
                raise ValueError("Can only query tensor or image!")

        temp = temp.to(self.device)

        # Next add augmentations
        if self._augmentations and self._training:
            temp = self._augmentations(temp)

        # Finally norm
        return self._transform(temp)

    def get_embedding(self, image: Tensor) -> Tensor:
        """Gets the embedding vector for the given image in (C, W, H). The image is assumed to be preprocessed."""
        image = image.to(self.device).unsqueeze(0)  # Make into batch shape
        with torch.no_grad():
            model = self.model
            model.eval()
            x = model.features(image)  # extracting feature
            x = model.avgpool(x)  # pooling
            x = torch.flatten(x, 1)
            # Getting and saving feature vector
            for i in range(3):
                x = model.classifier[i](x)

            return x.cpu().reshape(4096)


class ResNetModel(EmbeddingsModelImplementation):
    """Custom ResNet model for embedding extraction."""

    def embedding_size(self) -> int:
        return 2048

    def __init__(
        self,
        class_count: int,
        weights: Optional[Path] = None,
        pretrained: bool = False,
        device="cpu",
    ):
        """
        Wraps the torch model.
        :param class_count: Number of classes
        :param weights: Path to weights file to load
        :param pretrained: if weights are trained for transferware, or are general pretrained weights
        :param device: Device to load on
        """
        super().__init__()

        self.device = device
        # Use torch weights if we are using a pretrained weights and passed none
        model_resnet = torchvision.models.resnet50(
            weights=(
                torchvision.models.ResNet50_Weights.IMAGENET1K_V2
                if pretrained and weights is None
                else None
            )
        )

        # Loading our trained model, so need to change shape
        if not pretrained:
            model_resnet.fc = torch.nn.Linear(2048, class_count)

        # Load trained model if provided
        if weights:
            model_pth = torch.load(weights)
            model_resnet.load_state_dict(model_pth)

        # Add our class size
        if pretrained:
            model_resnet.fc = torch.nn.Linear(2048, class_count)

        self.model = model_resnet.to(device)

        # Returns the result of the last layer before the classifier
        self._extractor = create_feature_extractor(
            self.model, {"flatten": "flatten"}
        ).to(device)

        # Preprocessing steps
        self._transform = transforms.Compose(
            [
                transforms.Grayscale(3),
                transforms.Resize((224, 224)),
                torchvision.models.ResNet50_Weights.IMAGENET1K_V2.transforms().to(
                    device
                ),
            ]
        ).to(device)

    def transform(self, img: Tensor | Image) -> Tensor:
        """Preprocesses input, applying augmentations if in training mode."""
        # First convert to tensor
        match img:
            case Image():
                temp = transforms.ToTensor()(img)
            case Tensor():
                temp = img
            case _:
                raise ValueError("Can only query tensor or image!")

        temp = temp.to(self.device)

        # Next add augmentations
        if self._augmentations and self._training:
            temp = self._augmentations(temp)

        # Finally norm
        return self._transform(temp)

    def get_embedding(self, image: Tensor) -> Tensor:
        """Gets the embedding vector for the given image in (C, W, H). The image is assumed to be preprocessed."""
        image = image.to(self.device).unsqueeze(0)  # Make into batch shape
        with torch.no_grad():
            model = self._extractor
            model.eval()

            embedding = model.forward(image)["flatten"][0]

            return embedding.cpu()


class SwinModel(EmbeddingsModelImplementation):
    """Custom swin transformer model for embedding extraction."""

    def embedding_size(self) -> int:
        return 768

    def __init__(
        self,
        class_count: int,
        weights: Optional[Path] = None,
        pretrained: bool = False,
        device="cpu",
    ):
        """
        Wraps the torch model.
        :param class_count: Number of classes
        :param weights: Path to weights file to load
        :param pretrained: if weights are trained for transferware, or are general pretrained weights
        :param device: Device to load on
        """
        super().__init__()

        self.device = device
        # Use torch weights if we are using a pretrained weights and passed none
        model_swin = torchvision.models.swin_v2_t(
            weights=(
                torchvision.models.Swin_V2_T_Weights.IMAGENET1K_V1
                if pretrained and weights is None
                else None
            )
        )

        # Loading our trained model, so need to change shape
        if not pretrained:
            model_swin.head = torch.nn.Linear(768, class_count)

        # Load trained model if provided
        if weights:
            model_pth = torch.load(weights)
            model_swin.load_state_dict(model_pth)

        # Add our class size
        if pretrained:
            model_swin.head = torch.nn.Linear(768, class_count)

        self.model = model_swin.to(device)

        # Returns the result of the last layer before the classifier
        self._extractor = create_feature_extractor(
            self.model, {"flatten": "flatten"}
        ).to(device)

        # Preprocessing steps
        self._transform = transforms.Compose(
            [
                transforms.Grayscale(3),
                transforms.Resize((256, 256)),
                torchvision.models.Swin_V2_T_Weights.IMAGENET1K_V1.transforms().to(
                    device
                ),
            ]
        ).to(device)

    def transform(self, img: Tensor | Image) -> Tensor:
        """Preprocesses input, applying augmentations if in training mode."""
        # First convert to tensor
        match img:
            case Image():
                temp = transforms.ToTensor()(img)
            case Tensor():
                temp = img
            case _:
                raise ValueError("Can only query tensor or image!")

        temp = temp.to(self.device)

        # Next add augmentations
        if self._augmentations and self._training:
            temp = self._augmentations(temp)

        # Finally norm
        return self._transform(temp)

    def get_embedding(self, image: Tensor) -> Tensor:
        """Gets the embedding vector for the given image in (C, W, H). The image is assumed to be preprocessed."""
        image = image.to(self.device).unsqueeze(0)  # Make into batch shape
        with torch.no_grad():
            model = self._extractor
            model.eval()

            embedding = model.forward(image)["flatten"][0]

            return embedding.cpu()


class ConvnextModel(EmbeddingsModelImplementation):
    """Custom ConvNeXt model for embedding extraction."""

    def embedding_size(self) -> int:
        return 768

    def __init__(
        self,
        class_count: int,
        weights: Optional[Path] = None,
        pretrained: bool = False,
        device="cpu",
    ):
        """
        Wraps the torch model.
        :param class_count: Number of classes
        :param weights: Path to weights file to load
        :param pretrained: if weights are trained for transferware, or are general pretrained weights
        :param device: Device to load on
        """
        super().__init__()

        self.device = device
        # Use torch weights if we are using a pretrained weights and passed none
        model_cnext = torchvision.models.convnext_tiny(
            weights=(
                torchvision.models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1
                if pretrained and weights is None
                else None
            )
        )

        # Loading our trained model, so need to change shape
        if not pretrained:
            model_cnext.classifier[2] = torch.nn.Linear(768, class_count)

        # Load trained model if provided
        if weights:
            model_pth = torch.load(weights)
            model_cnext.load_state_dict(model_pth)

        # Add our class size
        if pretrained:
            model_cnext.classifier[2] = torch.nn.Linear(768, class_count)

        self.model = model_cnext.to(device)

        # Returns the result of the last layer before the classifier
        self._extractor = create_feature_extractor(
            self.model, {"classifier.1": "flattened_feat"}
        ).to(device)

        # Preprocessing steps
        self._transform = transforms.Compose(
            [
                transforms.Grayscale(3),
                transforms.Resize((224, 224)),
                torchvision.models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1.transforms().to(
                    device
                ),
            ]
        ).to(device)

    def transform(self, img: Tensor | Image) -> Tensor:
        """Preprocesses input, applying augmentations if in training mode."""
        # First convert to tensor
        match img:
            case Image():
                temp = transforms.ToTensor()(img)
            case Tensor():
                temp = img
            case _:
                raise ValueError("Can only query tensor or image!")

        temp = temp.to(self.device)
        # Next add augmentations
        if self._augmentations and self._training:
            temp = self._augmentations(temp)

        # Finally norm
        return self._transform(temp)

    def get_embedding(self, image: Tensor) -> Tensor:
        """Gets the embedding vector for the given image in (C, W, H). The image is assumed to be preprocessed."""
        image = image.to(self.device).unsqueeze(0)  # Make into batch shape
        with torch.no_grad():
            model = self._extractor
            model.eval()

            embedding = model.forward(image)["flattened_feat"][0]

            return embedding.cpu()


T = TypeVar("T", bound=EmbeddingsModelImplementation)


class ZhaoModel(Model):
    """
    Implementation of the query interface using an embeddings search approach.
    This class is the abstraction to the `EmbeddingsModelImplementation` implementation, in the GoF Bridge pattern.
    This class should handle the interactions between the vector store and underlying model, whereas that implementation
    handles the low level torch model details.
    """

    def __init__(
        self, resource_dir: Path, device, implementation_class: Type[T]
    ) -> None:
        super().__init__()
        self.resource_dir = resource_dir
        self.device = device
        self._implementation_class = implementation_class

        # All paths to resources
        idx_path = self.resource_dir.joinpath("zhao_index.ann").absolute()
        mappings_path = self.resource_dir.joinpath("zhao_index_mappings.pkl").absolute()
        cnt_path = self.resource_dir.joinpath("zhao_class_count.pkl").absolute()
        model_path = self.resource_dir.joinpath("zhao_train.pth").absolute()
        self.resources = [idx_path, mappings_path, cnt_path, model_path]

        self.class_count: int = torch.load(cnt_path)
        self.annoy_id_to_pattern_id: list[int] = torch.load(mappings_path)

        logging.debug(f"Loading model implementation: {implementation_class.__name__}")
        self.model: T = self._implementation_class(
            self.class_count, model_path, False, device
        )

        logging.debug(f"Loading annoy index")
        self.index = annoy.AnnoyIndex(self.model.embedding_size(), metric="euclidean")
        self.index.load(str(idx_path))

    def query(self, image: Tensor | Image, top_k: int = 10) -> list[ImageMatch]:
        with torch.no_grad():
            # Preprocess
            image = self.model.transform(image)
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

    def get_resource_files(self) -> list[Path]:
        return self.resources

    def make_tensorboard_projection(self, data: CacheDataset, sample_size: int):
        """Creates an embeddings projection for tensorflow with a certain sample size."""
        writer = SummaryWriter(
            log_dir=str(self.resource_dir.joinpath("tensorboard_logs").absolute())
        )
        data.set_transforms(self.model.transform)
        tensors = []
        embeddings = []

        sampler = RandomSampler(data, num_samples=sample_size)

        for i in tqdm(sampler):
            x, _ = data[i]
            x = x.to(self.device)
            embedding = self.model.get_embedding(x)

            embeddings.append(embedding)
            tensors.append(x)

        biiiig_img = torch.stack(tensors)
        biiiig_emb = torch.stack(embeddings)

        logging.debug("Writing embeddings to tensorboard")
        writer.add_embedding(biiiig_emb, label_img=biiiig_img, global_step=0)


class ZhaoTrainer(Trainer):
    def __init__(self, outer_dataset: Path, implementation_class: Type[T], device: str):
        super().__init__()
        self._outer_dataset = outer_dataset
        self._implementation_class = implementation_class
        self._device = device

    def train(self, dataset: CacheDataset) -> Model:
        logging.debug("Entering Zhao trainer")

        device = self._device
        model_wrapper = self._implementation_class(
            dataset.class_num(), None, True, device
        )
        augmentations = transforms.Compose(
            [
                transforms.ColorJitter(0.4, 0.4),
                transforms.RandomPerspective(),
                transforms.RandomResizedCrop((224, 224), scale=(0.05, 0.6)),
                transforms.RandomRotation(110, fill=(255, 255, 255)),
            ]
        ).to(device)
        model_wrapper.add_augmentations(augmentations)
        dataset.set_transforms(model_wrapper.transform)

        global_step = 0

        # Setting learning rate
        lr = 1e-5
        # Training the model for certain number of epochs
        epochs = settings.training.epochs

        model = model_wrapper.model

        # Using Adam optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        # Defining the loss function
        criteon = torch.nn.CrossEntropyLoss()

        # Split into test and train
        test_set, train_set = create_test_train_split(dataset, test_size=0.3)

        test_dataloader = DataLoader(
            test_set,
            batch_size=settings.training.batch_size,
            shuffle=False,
            num_workers=10,
            pin_memory=device != "cpu",
            pin_memory_device=device,
        )
        train_dataloader = DataLoader(
            train_set,
            batch_size=settings.training.batch_size,
            shuffle=True,
            num_workers=10,
            pin_memory=device != "cpu",
            pin_memory_device=device,
        )

        logging.debug("Data prepared, starting training")

        writer = SummaryWriter(
            str(self._outer_dataset.joinpath("tensorboard_logs").absolute())
        )

        # Track val improvement for early stopping
        best_val_loss = 1e20
        num_not_improved = 0

        if not settings.training.update_index_only:
            # Training process begins
            for epoch in range(epochs):
                logging.debug(f"Starting epoch {epoch}")
                loss_sum = 0

                model_wrapper.training_mode()

                map_metric_eval = BinaryAveragePrecision(thresholds=10).to(device)
                acc_eval = BinaryAccuracy(threshold=0.8).to(device)

                cutmix = transforms.CutMix(num_classes=dataset.class_num())
                mixup = transforms.MixUp(num_classes=dataset.class_num())
                cutmix_or_mixup = transforms.RandomChoice([cutmix, mixup]).to(device)

                for step, (x, y) in tqdm(
                    enumerate(train_dataloader), total=len(train_dataloader)
                ):
                    # Migrating data to gpu
                    x, y = x.to(device), y.to(device)

                    # Slice and dice images together (making them multiclass)
                    x, y = cutmix_or_mixup(x, y)

                    # Turn on model training mode
                    model.train()
                    # Generating predictions
                    logits = model(x)
                    # Calculating losses
                    loss = criteon.forward(logits, y)
                    # Recording total losses
                    loss_sum = loss_sum + loss.detach()
                    # Optimizer gradient zeroed
                    optimizer.zero_grad()
                    # Back propagation of loss to obtain loss gradient
                    loss.backward()
                    # Optimizing model
                    optimizer.step()
                    # Recording global step count
                    global_step += 1

                    writer.add_scalar("train/loss", loss, global_step)

                loss_sum = loss_sum / len(train_dataloader)

                # Opening the model evaluation mode
                model.eval()
                model_wrapper.eval_mode()

                logging.debug("Train complete, starting test")

                loss_sum_eval = 0
                # Evaluating in test sets
                for step, (x, y) in tqdm(
                    enumerate(test_dataloader), total=len(test_dataloader)
                ):
                    x, y = x.to(device), y.to(device)

                    # Calculating model prediction results
                    logits = model(x)
                    # Calculating losses
                    loss = criteon.forward(logits, y)
                    loss_sum_eval = loss_sum_eval + loss.detach()

                    writer.add_scalar("val/loss", loss, (epoch + 1) * step)

                    with torch.no_grad():
                        # Create PR curve
                        probs = torch.softmax(logits, dim=1)
                        one_hot_labels = torch.nn.functional.one_hot(
                            y, num_classes=dataset.class_num()
                        )
                        writer.add_pr_curve(
                            "val/pr", one_hot_labels, probs, (epoch + 1) * step
                        )
                        writer.add_scalar(
                            "val/map",
                            map_metric_eval(probs, one_hot_labels),
                            (epoch + 1) * step,
                        )

                        acc_eval.update(probs, one_hot_labels)

                eval_acc = acc_eval.compute()
                writer.add_scalar("val/avg_accuracy", eval_acc, epoch)

                loss_sum_eval = loss_sum_eval / len(test_dataloader)

                writer.add_scalars(
                    "Train vs Test",
                    {"val": loss_sum_eval, "train": loss_sum},
                    global_step,
                )

                # Save model if improved
                if loss_sum_eval < best_val_loss:
                    logging.debug("Saving best model")
                    torch.save(
                        model.to("cpu").state_dict(),
                        self._outer_dataset.joinpath("zhao_train_best.pth"),
                    )
                    best_val_loss = loss_sum_eval
                    num_not_improved = 0
                else:
                    num_not_improved += 1

                    # Stop training if val keeps not improving
                    if num_not_improved > settings.training.early_stopping_thresh:
                        break

            # Rename best weights so they load automatically
            self._outer_dataset.joinpath("zhao_train_best.pth").rename(
                self._outer_dataset.joinpath("zhao_train.pth")
            )

        # There can be more classes in the dataset if only using index, so just load
        class_num = (
            torch.load(self._outer_dataset.joinpath("zhao_class_count.pkl").absolute())
            if settings.training.update_index_only
            else dataset.class_num()
        )

        # Reload best weights
        model_wrapper = self._implementation_class(
            class_num,
            self._outer_dataset.joinpath("zhao_train.pth"),
            pretrained=False,
            device=device,
        )

        logging.debug("Building vector store")
        index, idx_mappings = self.generate_annoy_cache(model_wrapper, dataset)

        logging.debug("Saving resources to disk")
        self.save_resources(class_num, idx_mappings, index)

        return ZhaoModel(
            resource_dir=self._outer_dataset,
            device=device,
            implementation_class=self._implementation_class,
        )

    def save_resources(
        self, class_cnt: int, idx_mappings: list[int], index: annoy.AnnoyIndex
    ):
        """Saves training resources to disk"""
        # Save to disk
        idx_path = self._outer_dataset.joinpath("zhao_index.ann").absolute()
        mappings_path = self._outer_dataset.joinpath(
            "zhao_index_mappings.pkl"
        ).absolute()
        cnt_path = self._outer_dataset.joinpath("zhao_class_count.pkl").absolute()

        index.save(str(idx_path))
        torch.save(idx_mappings, mappings_path)
        # We need class count later when loading the model on the api
        torch.save(class_cnt, cnt_path)

    def generate_annoy_cache(
        self,
        model: T,
        ds: CacheDataset,
        visitor: Optional[Callable] = None,
    ) -> tuple[annoy.AnnoyIndex, list[int]]:
        """
        Builds an annoy index for the dataset, using embeddings given by the model. Returns the index and a list of
        annoy index ids to pattern ids used in the dataset.
        """
        ds.set_transforms(model.transform)

        index = annoy.AnnoyIndex(model.embedding_size(), metric="euclidean")
        # Each index is the annoy id, each element is the matching tcc pattern id
        aid_to_tccid: list[int] = []

        pattern_ids = ds.get_pattern_ids()

        for i in tqdm(range(len(ds))):
            # Load image
            img, _ = ds[i]
            img = img.to(model.device)
            pattern_id = pattern_ids[i]

            # Extract embedding
            embedding = model.get_embedding(img)
            # Add vector to cache
            index.add_item(i, embedding.detach())
            aid_to_tccid.append(pattern_id)

            if visitor:
                visitor(embedding, img, i)

        index.build(10000)
        return index, aid_to_tccid


class EmbeddingsValidator(GenericValidator):
    """Simple wrapper around the generic validator that adds transforms from embedding models."""

    def validate(
        self, model: ZhaoModel, validation_set: ImageFolder
    ) -> tuple[dict[int, float], float]:
        validation_set.transform = model.model.transform

        return super().validate(model, validation_set)


class ZhaoModelFactory(AbstractModelFactory):
    def __init__(self, resource_path: Path, device: str):
        super().__init__(resource_path)
        # Underlying torch class wrapper used, allows for swapping model backends
        self._implementation_class = ConvnextModel
        self._device = device

    def get_model(self) -> ZhaoModel:
        return ZhaoModel(
            self._resource_path,
            self._device,
            self._implementation_class,
        )

    def get_trainer(self) -> ZhaoTrainer:
        return ZhaoTrainer(
            self._resource_path, self._implementation_class, self._device
        )

    def get_validator(self) -> EmbeddingsValidator:
        return EmbeddingsValidator(self._device)
