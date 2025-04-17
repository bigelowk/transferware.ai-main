from typing import Any

from pydantic import BaseModel, field_validator


class ImageMatch(BaseModel):
    """An image matching to the query image/"""

    id: int
    """Id of matching image"""
    confidence: float
    """Confidence metric of matching image. Could be 0-1, or a distance, depending on model."""

    def model_post_init(self, __context: Any) -> None:
        # change from angular distance to percentage similarity
        self.confidence = (1 - (self.confidence / 2)) * 100

    # @field_validator('confidence')
    # def angular_to_percentage(cls, v):
    #     return (1 - (v / 2)) * 100






img = ImageMatch(id = 101, confidence = 0)
print(img.confidence)