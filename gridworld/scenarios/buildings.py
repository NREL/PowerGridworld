import pandas as pd

from gridworld import MultiComponentEnv
from gridworld import MultiAgentEnv
from gridworld.agents.buildings import FiveZoneROMThermalEnergyEnv
from gridworld.agents.pv import PVEnv
from gridworld.agents.energy_storage import EnergyStorageEnv
from gridworld.distribution_system import OpenDSSSolver


def make_env_config(
        building_config=None,
        pv_config=None,
        storage_config=None,
        system_load_rescale_factor=0.65,
        num_buildings=3):

    components = [
        {
            "name": "building",
            "cls": FiveZoneROMThermalEnergyEnv,
            "config": building_config
        }
    ]

    if pv_config is not None:
        components.append({
            "name": "pv",
            "cls": PVEnv,
            "config": pv_config
        })

    if storage_config is not None:
        components.append({
            "name": "storage",
            "cls": EnergyStorageEnv,
            "config": storage_config
        })

    common_config = {
        "start_time": "08-12-2021 00:00:00",
        "end_time": "08-13-2021 00:00:00",
        "control_timedelta": pd.Timedelta(300, "s")
    }

    agents = [
        {
            "name": "building-{}".format(i),
            "bus": "675c",
            "cls": MultiComponentEnv,
            "config": {"components": components}
        } for i in range(num_buildings)
    ]

    # OpenDSS configuration.  Note that file paths are relative to 
    # "power_gridworld/distribution_system/data" by default.
    pf_config = {
        "cls": OpenDSSSolver,
        "config": {
            "feeder_file": "ieee_13_dss/IEEE13Nodeckt.dss",
            "loadshape_file": "ieee_13_dss/annual_hourly_load_profile.csv",
            "system_load_rescale_factor": system_load_rescale_factor,
        }
    }

    env_config = {
        "common_config": common_config,
        "pf_config": pf_config,
        "agents": agents
    }

    return env_config

