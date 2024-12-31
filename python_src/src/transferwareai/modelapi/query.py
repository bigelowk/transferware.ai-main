import asyncio
import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated
import tarfile

from aiomqtt import Client
from filelock import Timeout, FileLock

import torchvision
from fastapi import FastAPI, File, Header, UploadFile, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
import torch
from pydantic import BaseModel

from transferwareai.config import settings
from transferwareai.modelapi.model import (
    initialize_model,
    get_model,
    get_api,
    reload_api_cache,
)
from transferwareai.modelapi.mqtt import mqtt_sub_process
from transferwareai.models.adt import ImageMatch, Model
from transferwareai.tccapi.api_cache import ApiCache
from fastapi.responses import FileResponse

# Required to avoid GC collecting tasks
background_tasks = set()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Runs startup and shutdown code"""
    # Load caches and models
    await initialize_model()
    # Start listening for reload commands in the background
    back_client = asyncio.create_task(mqtt_sub_process())
    back_client.add_done_callback(background_tasks.discard)
    background_tasks.add(back_client)
    yield


app = FastAPI(lifespan=lifespan)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.query.origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/query", response_model=list[ImageMatch])
async def query_model(
    file: Annotated[bytes, File()], model: Annotated[Model, Depends(get_model)]
):
    """Send an image to the model, and get the 10 closest images back."""

    start = time.clock_gettime_ns(time.CLOCK_MONOTONIC)

    # Parse image into tensor
    try:
        raw_tensor = torch.frombuffer(file, dtype=torch.uint8)
        img = torchvision.io.decode_image(raw_tensor, torchvision.io.ImageReadMode.RGB)
    except Exception as e:
        raise HTTPException(
            status_code=400, detail="Only jpg or png files are supported"
        )

    # Query model
    top_matches = model.query(img, top_k=settings.query.top_k)

    end = time.clock_gettime_ns(time.CLOCK_MONOTONIC)

    logging.debug(f"Query took {(end - start) / 1e6}ms.")

    return top_matches


@app.get("/pattern/image/{id}")
async def get_image_for_id(id: int, api: Annotated[ApiCache, Depends(get_api)]):
    """Gets the main image for a pattern ID."""
    p = api.get_image_file_path_for_tag(id, "pattern")
    return FileResponse(p)


class Metadata(BaseModel):
    pattern_id: int
    pattern_name: str
    tcc_url: str


@app.get("/pattern/{id}", response_model=Metadata)
async def get_data_for_pattern(id: int, api: Annotated[ApiCache, Depends(get_api)]):
    """Gets metadata for a pattern ID."""
    name = api.get_name_for_pattern_id(id)
    url = api.get_tcc_url_for_pattern_id(id)

    return Metadata(pattern_id=id, pattern_name=name, tcc_url=url)


@app.post("/update")
async def update_model(file: UploadFile, token=Header("Authorization")):
    """Uploads new model resources. file is a tar archive that will be extracted into resource directory."""
    # Verify access token
    if token != settings.access_token:
        raise HTTPException(status_code=401, detail="Invalid access token")

    lock_path = Path(settings.query.resource_dir) / ".$model.lock"

    # Acquire lock
    try:
        with FileLock(lock_path, timeout=0):
            # Extract the tarball
            with tarfile.open(fileobj=file.file) as t:
                logging.debug("Extracting new model resources")
                t.extractall(path=settings.query.resource_dir)

            # Update the api cache since model may use it
            await reload_api_cache(update=True)
    except Timeout:
        raise HTTPException(status_code=503, detail="Model update in progress")

    # Pub message telling nodes to reload their resources (including this one)
    # spawn a new client each time because we will rarely update anyhow
    async with Client("broker") as client:
        logging.debug("Pub reload")
        await client.publish("transferwareai/reload", qos=2)
