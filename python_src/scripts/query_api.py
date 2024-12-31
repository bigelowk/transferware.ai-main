# Runs the query api
import logging
import warnings

import uvicorn

from transferwareai.config import settings

# Set loggers to not spam
warnings.filterwarnings("ignore", ".*Tensor.*")
logging.getLogger("PIL").setLevel(logging.WARN)
logging.getLogger("filelock").setLevel(logging.WARN)
logging.getLogger("matplotlib").setLevel(logging.WARN)
logging.getLogger("multipart").setLevel(logging.WARN)

logging.getLogger().setLevel(logging.DEBUG)

if __name__ == "__main__":
    uvicorn.run(
        "transferwareai.modelapi.query:app",
        host=settings.query.host,
        port=settings.query.port,
        workers=settings.query.workers,
        loop="asyncio",
        root_path="/api"
    )
