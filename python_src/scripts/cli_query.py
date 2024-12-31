# Script to view the closest images from the CLI
import argparse
import logging
import os
import re
import shutil
import warnings
from pathlib import Path

import torch
from tqdm import tqdm

from transferwareai.data.dataset import CacheDataset
from transferwareai.models.construct import get_abstract_factory
import torchvision.transforms.functional as F
from transferwareai.config import settings
from transferwareai.tccapi.api_cache import ApiCache
import torchvision
import time
import matplotlib.pyplot as plt
import requests
from io import BytesIO

# Set loggers to not spam
warnings.filterwarnings("ignore", ".*Tensor.*")
logging.getLogger("PIL").setLevel(logging.WARN)
logging.getLogger("filelock").setLevel(logging.WARN)
logging.getLogger("matplotlib").setLevel(logging.WARN)

logging.getLogger().setLevel(logging.DEBUG)

if __name__ == "__main__":
    # Args
    parser = argparse.ArgumentParser(
        prog="cli_query",
        description="Queries the TCC database on the CLI",
    )
    parser.add_argument("input", help="Url to image or path to image to query with")
    args = parser.parse_args()

    # Regex for finding URL
    is_url = re.compile(
        r"^(?:http|ftp)s?://"  # http:// or https://
        r"(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)|"  # domain...
        r"localhost|"  # localhost...
        r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})"  # ...or ip
        r"(?::\d+)?"  # optional port
        r"(?:/?|[/?]\S+)$",
        re.IGNORECASE,
    )

    res_path = Path(settings.query.resource_dir)

    # Get model
    factory = get_abstract_factory(settings.model_implimentation, "query")
    model = factory.get_model()

    # Get cache
    api = ApiCache.from_cache(res_path.joinpath("cache"), no_update=True)
    ds = CacheDataset(api, skip_ids=settings.training.skip_ids)

    # Get image
    if is_url.match(args.input):
        logging.info("Detected URL, downloading image...")

        # Get image from URL
        r = requests.get(args.input, stream=True)
        if r.status_code == 200:
            # Decompress
            r.raw.decode_content = True
            b = BytesIO()
            shutil.copyfileobj(r.raw, b)

            # decode image
            raw_tensor = torch.frombuffer(b.getbuffer(), dtype=torch.uint8)
            img = torchvision.io.decode_image(
                raw_tensor, torchvision.io.ImageReadMode.RGB
            )
        else:
            logging.error("Url did not return image!")
            exit(1)
    else:
        logging.info("Detected path, reading from disk...")
        # Path is file on disk
        img_path = Path(args.input)
        img = torchvision.io.read_image(
            str(img_path.absolute()), torchvision.io.ImageReadMode.RGB
        )

    # Get matches
    start = time.clock_gettime_ns(time.CLOCK_MONOTONIC)
    top = model.query(img)
    end = time.clock_gettime_ns(time.CLOCK_MONOTONIC)

    print(f"Ran in {(end-start)/1e+6:.2f}ms")

    print(top)

    logging.info("Generating image...")

    close_images = []
    cache_path = res_path.joinpath("cache/assets")

    # Build tensor that just concat all the images of matches
    for m in tqdm(top):
        d = api.get_image_path_for_id(m.id)
        files = os.listdir(d)

        for file in files:
            try:
                close_images.append(
                    F.resize(
                        torchvision.io.read_image(
                            str(d.joinpath(file)),
                            torchvision.io.ImageReadMode.RGB,
                        ),
                        size=[500, 500],
                    )
                )
            except RuntimeError as e:
                logging.error(f"Error reading files: {e}")

    grid = torchvision.utils.make_grid(close_images)
    plt.imshow(grid.permute(1, 2, 0))

    # Display to user
    plt.show()
