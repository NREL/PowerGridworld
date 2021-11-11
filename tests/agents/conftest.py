import pytest


@pytest.fixture
def building_config():
    return {
        "start_time": "08-12-2020 00:00:00",
        "end_time": "08-13-2020 00:00:00",
    }


@pytest.fixture
def pv_config():
    return {
        "name": "pv",
        "profile_csv": "pv_profile.csv",
        "scaling_factor": 10.
    }


@pytest.fixture
def pv_array_config():
    return {
        "name": "pv",
        "profile_csv": "pv_profile.csv",
        "scaling_factor": 400.
    }


@pytest.fixture
def ev_charging_config():
    return {
        "num_vehicles": 100,
        "minutes_per_step": 5,
        "max_charge_rate_kw": 7.,
        "peak_threshold": 250.,
        "vehicle_multiplier": 5.,
        "rescale_spaces": False
    }


@pytest.fixture
def energy_storage_config():
    return {}

    


