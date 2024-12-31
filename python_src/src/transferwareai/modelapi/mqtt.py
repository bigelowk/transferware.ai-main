import logging
from aiomqtt import Client
from transferwareai.modelapi.model import reload_model, reload_api_cache


async def mqtt_sub_process():
    """Spawns a long-running MQTT client that handles reload requests to the server"""
    logging.debug("Entering background process")
    async with Client("broker") as client:
        # Sub to an MQTT topic signaled whenever this api should reload
        await client.subscribe("transferwareai/#", qos=2)
        logging.debug("Subscribed to reload MQTT")

        async for msg in client.messages:
            # If we are here, no one should be writing to the files, so we don't need to lock
            logging.debug("Reloading model from disk")
            # These lock resources, so we don't race the api here
            await reload_model()
            await reload_api_cache(update=False)
            logging.debug("Finished reloading model")
