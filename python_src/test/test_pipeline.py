import PIL.Image
import pytest
import torchvision.io

from transferwareai.models.adt import Model
from transferwareai.models.construct import get_abstract_factory


class TestPipeline:

    @pytest.fixture(scope="class")
    def model(self) -> Model:
        fact = get_abstract_factory("ZhaoModelFactory", "query")
        model = fact.get_model()
        return model

    def test_tensor(self, model: Model):
        t = torchvision.io.read_image(
            "./assets/37742-pattern.jpg", torchvision.io.ImageReadMode.RGB
        )
        # Normal amount
        res = model.query(t, top_k=10)

        assert len(res) == 10

        # More
        res = model.query(t, top_k=20)

        assert len(res) == 20

        # Make sure image is actually in response
        assert 37742 in [r.id for r in res]

    def test_pil(self, model: Model):
        # Test PIL input as well
        im = PIL.Image.open("./assets/37742-pattern.jpg")
        res = model.query(im)

        assert len(res) == 10
