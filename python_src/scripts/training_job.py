# Training job script to be run at some interval to update the model and deploy it to the query api. All of its actual
# Implementation should be in the main module.
from transferwareai.config import settings
from transferwareai.training_job import TrainingJob
import logging
import warnings

# Set loggers to not spam
warnings.filterwarnings("ignore", ".*Tensor.*")
logging.getLogger("PIL").setLevel(logging.WARN)

logging.getLogger().setLevel(logging.DEBUG)

if __name__ == "__main__":
    job = TrainingJob()
    job.exec(update_cache=settings.update_cache)
