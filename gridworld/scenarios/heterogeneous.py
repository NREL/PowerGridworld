from os import system
import pandas as pd

from gridworld import MultiComponentEnv
from gridworld import MultiAgentEnv
from gridworld.agents.buildings import FiveZoneROMThermalEnergyEnv
from gridworld.agents.pv import PVEnv
from gridworld.agents.energy_storage import EnergyStorageEnv
from gridworld.agents.vehicles import EVChargingEnv
from gridworld.distribution_system import OpenDSSSolver


def make_env_config(system_load_rescale_factor=0.65, rescale_spaces=True):

    # Make the multi-component building
    building_components = [
        {
            "name": "building",
            "cls": FiveZoneROMThermalEnergyEnv,
            "config": {
                "reward_structure": {"alpha": 0.0},  # all comfort focused
                "rescale_spaces": rescale_spaces
            }
        }
    ]
    building_components.append({
        "name": "pv",
        "cls": PVEnv,
        "config": {
            "profile_csv": "off-peak.csv",
            "scaling_factor": 40.,
            "rescale_spaces": rescale_spaces
        }
    })
    building_components.append({
        "name": "storage",
        "cls": EnergyStorageEnv,
        "config": {
            "max_power": 20.,
            "storage_range": (3., 250.),
            "rescale_spaces": rescale_spaces
        } 
    })

    # PV farm agent gets rewarded stabilizing the bus voltage.
    class ThisPVEnv(PVEnv):
        def step_reward(self, **kwargs):
            v = kwargs["min_voltage"]
            viol_lower = min(0, v - 0.95)
            viol_upper = min(0, 1.05 - v)
            viol = viol_lower + viol_upper
            return -(1000*viol)**2, {}

    # Common configuration
    common_config = {
        "start_time": "08-12-2020 00:00:00",
        "end_time": "08-13-2020 00:00:00",
        "control_timedelta": pd.Timedelta(300, "s"),
    }

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

    # List of agents for multiagent env constructor.
    agents = [
        {
            "name": "building",
            "bus": "675c",
            "cls": MultiComponentEnv,
            "config": {"components": building_components}
        },
        {
            "name": "pv",
            "bus": "675c",
            "cls": ThisPVEnv,
            "config": {
                "profile_csv": "constant.csv",
                "scaling_factor": 400.,
                "rescale_spaces": rescale_spaces,
                "grid_aware": True
            }
        },
        {
            "name": "ev-charging",
            "bus": "675c",
            "cls": EVChargingEnv,
            "config": {
                "num_vehicles": 25,
                "minutes_per_step": 5,
                "max_charge_rate_kw": 7.,
                "peak_threshold": 200.,
                "vehicle_multiplier": 40.,
                "rescale_spaces": rescale_spaces
            }
        }
    ]

    env_config = {
        "common_config": common_config,
        "pf_config": pf_config,
        "agents": agents
    }

    return env_config
