import pytest


@pytest.fixture(scope="function")
def opendss_config():
    
    return {
        "feeder_file": "ieee_13_dss/IEEE13Nodeckt.dss",
        "loadshape_file": "ieee_13_dss/annual_hourly_load_profile.csv",
        "system_load_rescale_factor": 0.7
    }

