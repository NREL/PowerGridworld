import pytest


@pytest.fixture(scope="function")
def building_config():
    return {
        "start_time": "08-12-2020 00:00:00",
        "end_time": "08-13-2020 00:00:00",
    }
