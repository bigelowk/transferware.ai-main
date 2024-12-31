import pytest
from transferwareai.tccapi.api_cache import ApiCache


class TestApi:

    @pytest.mark.parametrize("num", (1, 2, 100))
    def test_page(self, num):
        assert ApiCache.get_api_page(num) is not None

    def test_end_page(self):
        assert ApiCache.get_api_page(999999999) == []
