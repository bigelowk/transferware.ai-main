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

from fastapi import Request
from pymongo import MongoClient
from datetime import datetime
import os
import base64
from fastapi.responses import JSONResponse

# Required to avoid GC collecting tasks
background_tasks = set()
result_id = None

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

#mongoDB set up

mongo_uri = os.environ.get('MONGO_URI', 'mongodb://localhost:27017/SurveyApp')
client = MongoClient(mongo_uri)
db = client.get_default_database()

@app.post("/query", response_model=list[ImageMatch])
async def query_model(
    request: Request,
    #file: Annotated[bytes, File()], model: Annotated[Model, Depends(get_model)],
    file: UploadFile, model: Annotated[Model, Depends(get_model)],
    #file: Annotated[bytes, File()]
):
    global result_id
    """Send an image to the model, and get the 10 closest images back."""

    start = time.clock_gettime_ns(time.CLOCK_MONOTONIC)

# Get Client IP Address
    client_ip = request.client.host

    # Capture submission timestamp
    submission_time = datetime.utcnow()

    # Read the uploaded file as binary
    image_data = await file.read()

    # Convert image to Base64 string
    image_base64 = base64.b64encode(image_data).decode("utf-8")

    logging.info(f"Image submitted from IP: {client_ip} at {submission_time}")

    # Parse image into tensor
    try:
        #raw_tensor = torch.frombuffer(file, dtype=torch.uint8)
        raw_tensor = torch.frombuffer(file, dtype=torch.uint8)
        img = torchvision.io.decode_image(raw_tensor, torchvision.io.ImageReadMode.RGB)
    except Exception as e:
        raise HTTPException(
            status_code=400, detail="Only jpg or png files are supported"
        )

    # Query model
    top_matches = model.query(img, top_k=settings.query.top_k + 10)

    
    end = time.clock_gettime_ns(time.CLOCK_MONOTONIC)
    query_time = (end - start) / 1e6  # Convert to milliseconds

    logging.debug(f"Query took {(end - start) / 1e6}ms.")

    # take out the repeated results
    cleaned_matches = []
    ids = []
    for image in top_matches:
        if len(cleaned_matches) < settings.query.top_k:
            if image.id not in ids:
                cleaned_matches.append(image)
                ids.append(image.id)
        if len(cleaned_matches) >= settings.query.top_k:
            break
    
    matches = {}
    for img in cleaned_matches:
        matches[str(img.id)] = img.confidence

    # Store submission details in MongoDB
    #collection = mongoClient()
    result = db.image_analytics.insert_one({
        "submission_time": submission_time,
        "query_time_ms": query_time,
        "client_ip": client_ip,  # Store IP in DB
        "image_filename": file.filename,
        "image_base64": image_base64,  # Store the encoded image
        "confidence_intervals": matches
    })

    result_id = str(result.inserted_id)
    
    logging.info(f"Query from {client_ip} {result_id} took {query_time}ms.")

    return cleaned_matches


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
