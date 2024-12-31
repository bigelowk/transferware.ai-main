from typing import Optional, Any
from itertools import chain

import requests
from pathlib import Path
import polars as pl
import json
import os
import logging

from polars import DataFrame

import asyncio
import aiohttp
from aiofile import async_open

import nest_asyncio

nest_asyncio.apply()


class ApiCache:
    """Class to manage API cache"""

    @staticmethod
    def api_url(num: int | None = 1) -> str:
        """Get the TCC API URL for a given page number"""
        return (
            f"https://db.transferwarecollectorsclub.org/api/v1/patterns/page{num}?key=umich2024"
            if num > 1
            else "https://db.transferwarecollectorsclub.org/api/v1/patterns/?key=umich2024"
        )

    @staticmethod
    def get_api_page(page: int) -> Optional[list[dict[str, Any]]]:
        """
        Retrieve data from a single API page.
        :param page: page number to get.
        :return: list of json patterns.
        """
        response = requests.get(ApiCache.api_url(page))
        if response.status_code != 200:
            return None
        return response.json()

    @staticmethod
    def get_api_pages(limit: None | int = None) -> list[dict[str, Any]]:
        """
        Get all patterns from API.
        :param limit: Limit the number of pages to retrieve.
        :return list of json pattern objects.
        """
        patterns = []
        page_num = 1
        # Ends once get_api_page returns [] or error getting page
        while page := ApiCache.get_api_page(page_num):
            logging.debug(f"Retrieved page {page_num}")

            # Last page
            if len(page) == 0:
                break

            patterns.extend(page)
            page_num += 1

            if limit and page_num > limit:
                break
        return patterns

    @staticmethod
    async def get_api_page_async(
        page: int, session: aiohttp.ClientSession
    ) -> Optional[list[dict[str, Any]]]:
        """
        Retrieve data from a single API page.
        :param session: HTTP session to use.
        :param page: page number to get.
        :return: list of json patterns.
        """
        async with session.get(ApiCache.api_url(page)) as response:
            if response.status != 200:
                return None
            return await response.json()

    @staticmethod
    async def get_api_pages_async() -> list[dict[str, Any]]:
        """
        Get all patterns from API.
        :return list of json pattern objects.
        """
        patterns = []
        page_num = 0
        batch_size = 20
        done = False

        async with aiohttp.ClientSession() as session:
            while not done:
                # Spawn page gets in batches
                tasks = []
                for page in range(page_num + 1, page_num + batch_size + 1):
                    tasks.append(
                        asyncio.create_task(ApiCache.get_api_page_async(page, session))
                    )
                page_num += batch_size
                logging.debug(f"Downloading up to {page_num}")

                results = await asyncio.gather(*tasks)
                patterns.extend(
                    (
                        pattern
                        for pattern in chain.from_iterable(results)
                        if pattern is not None
                    )
                )

                # End if nothing is received
                if [] in results:
                    done = True

        return patterns

    def __init__(self, directory: Path, df: DataFrame):
        self._directory = directory
        self._cache_file = directory.joinpath("cache.json")
        self._assets_dir = directory.joinpath("assets")  # directory for images
        self._df = df

    @staticmethod
    def _requires_update(cache_file: Path) -> bool:
        if cache_file.exists():
            df = pl.read_json(cache_file)
            max_id_cache = df["id"].max()
            max_id_now = ApiCache.get_api_page(1)[0]["id"]

            return max_id_cache < max_id_now
        else:
            return True

    @staticmethod
    def from_cache(directory: Path, no_update: bool = False) -> "ApiCache":
        """Reads the cache in directory, ensures it is up-to-date, and then returns the wrapper object."""

        _directory = directory
        _cache_file = directory.joinpath("cache.json")
        _assets_dir = directory.joinpath("assets")  # directory for images

        runner = asyncio.get_event_loop()

        if not no_update and ApiCache._requires_update(_cache_file):
            logging.info("Cache out of date, updating cache")

            # Get patterns JSON
            patterns = runner.run_until_complete(ApiCache.get_api_pages_async())

            if not _directory.exists():
                os.makedirs(_directory)

            # Write cache to disk
            with open(_cache_file, "w") as buffer:
                buffer.write(json.dumps(patterns, indent=2))

        df = pl.read_json(_cache_file)

        logging.info(f"Loaded cache with {len(df)} patterns")

        # Early return if not updating
        if no_update:
            return ApiCache(directory, df)

        # Begin collecting assets
        if not _assets_dir.exists():
            os.makedirs(_assets_dir)  # create assets directory

        # Query for pattern ids + image URLs and tags
        urls = df.select(
            pl.col("id"),
            pl.col("images").list.eval(pl.element().struct.field("url")),
            pl.col("images").list.eval(pl.element().struct.field("tags")).alias("tags"),
        ).drop_nulls()

        # We use asyncio for downloading to avoid it taking like years
        async def get_images_for_pattern(
            pattern_id, image_urls, tags, pattern_dir, session
        ):
            """Download images for a pattern and write them to disk."""
            logging.debug(f"Downloading images for {pattern_id}")
            # Download all images for pattern
            for image_url, tag in zip(image_urls, tags):
                # Make path safe
                tag = tag.replace("/", "_")

                if not pattern_dir.joinpath(f"{pattern_id}-{tag}.jpg").exists():
                    complete = False
                    while not complete:
                        try:
                            async with session.get(image_url) as image_response:
                                image_file = pattern_dir.joinpath(
                                    f"{pattern_id}-{tag}.jpg"
                                )
                                async with async_open(image_file, "wb") as f:
                                    await f.write(await image_response.content.read())
                            complete = True
                        except aiohttp.ServerDisconnectedError as e:
                            logging.debug("Server disconnect! Reattempting")

        async def get_images():
            # Share a single connection pool
            async with aiohttp.ClientSession(
                connector=aiohttp.TCPConnector(limit=200)
            ) as session:
                tasks = []
                # Create download image tasks, lazy
                for row in urls.iter_rows():
                    pattern_id, image_urls, tags = row

                    # Create patterns directory if new
                    pattern_dir = _assets_dir.joinpath(str(pattern_id))
                    if not pattern_dir.exists():
                        os.makedirs(pattern_dir)

                    tasks.append(
                        asyncio.create_task(
                            get_images_for_pattern(
                                pattern_id, image_urls, tags, pattern_dir, session
                            )
                        )
                    )
                # Download each pattern in parallel, and each image per pattern in series
                await asyncio.gather(*tasks)

        # Actually run the image get tasks
        runner.run_until_complete(get_images())

        return ApiCache(directory, df)

    def as_df(self) -> DataFrame:
        return self._df

    def get_image_path_for_id(self, pattern_id) -> Path:
        """Returns path to image directory for a given id."""
        return self._assets_dir.joinpath(str(pattern_id))

    def get_image_file_path_for_tag(self, pattern_id, tag) -> Path:
        """Returns path to a specific image file given id and tag."""
        tag = tag.replace("/", "_")
        return self.get_image_path_for_id(pattern_id).joinpath(
            f"{pattern_id}-{tag}.jpg"
        )

    def get_name_for_pattern_id(self, pattern_id: int) -> str:
        return self._df.select(pl.col("name").where(pl.col("id") == pattern_id))[
            "name"
        ][0]

    def get_tcc_url_for_pattern_id(self, pattern_id: int) -> str:
        return self._df.select(pl.col("url").where(pl.col("id") == pattern_id))["url"][
            0
        ]
