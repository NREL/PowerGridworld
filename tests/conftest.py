import pytest

import pandas as pd

from tests.distribution_system.conftest import opendss_config

import gymnasium as gym

from gridworld.agents.buildings import FiveZoneROMThermalEnergyEnv
from gridworld.agents.pv import PVEnv
from gridworld.agents.energy_storage import EnergyStorageEnv
from gridworld import MultiAgentEnv, MultiComponentEnv
from gridworld.distribution_system import OpenDSSSolver


## Functions for running simply policy baselines ## 
def policy(space, kind="low"):
    """Returns an action of the specified kind given the action space."""
    
    assert kind in ["low", "high", "random"], f"invalid policy kind='{kind}'"

    if isinstance(space, (gym.spaces.Box, gym.spaces.Discrete)):

        if kind == "low":
            return space.low
        elif kind == "high":
            return space.high
        else: 
            return space.sample()

    elif isinstance(space, gym.spaces.Dict):

        return {name: policy(space[name], kind=kind) for name in space}

    else:

        raise ValueError(f"unsupported space {type(space)}")


def single_agent_single_episode_runner(env, kind):
    """Runs an episode with a single min/max/random policy, returning True if
    successful."""
    
    env.reset()
    done = False
    while not done:
        space = env.action_space
        action = policy(space, kind)
        _, _, done, _ = env.step(action)
    
    return True


def single_agent_episode_runner(env):
    """Runs an environment using min/max/random policies."""

    results = [
        single_agent_single_episode_runner(env, kind) 
            for kind in ["low", "high", "random"]
    ]
    
    return all(results)


def multi_agent_single_episode_runner(env, kind):
    """Runs a single episode of multiagent env using policy type 
    high/low/random."""

    if isinstance(env, MultiComponentEnv):
        done = False
        def is_done(done):
            return done
    elif isinstance(env, MultiAgentEnv):
        done = {"__all__": False}
        def is_done(done):
            return done["__all__"]
    else:
        raise TypeError(f"env type {type(env)} is not supported for this test")

    env.reset()
    while not is_done(done):
        action = {name: policy(space, kind) for name, space in env.action_space.items()}
        _, _, done, _ = env.step(action)

    return True


def multi_agent_episode_runner(env):
    """Run a multiagent env for episodes using the different policy types."""

    results = [
        multi_agent_single_episode_runner(env, p) for p in ["low", "high", "random"]
    ]

    return all(results)
    

## Fixtures for env creation ##

@pytest.fixture(scope="function")
def common_config():
    return {
        "start_time": "08-12-2020 00:00:00",
        "end_time": "08-13-2020 00:00:00",
        "control_timedelta": pd.Timedelta(300, "s")
    }


@pytest.fixture(scope="function")
def pf_config(opendss_config):
    return {
        "cls": OpenDSSSolver,
        "config": opendss_config
    }


@pytest.fixture(scope="function")
def multicomponent_building_config():

    building = {
        "name": "building",
        "cls": FiveZoneROMThermalEnergyEnv,
        "config": {
            "start_time": "08-12-2020 00:00:00",
            "end_time": "08-13-2020 00:00:00",
            "rescale_spaces": False,
            "obs_config": {
                "zone_temp": (18, 34),
                "p_consumed": (-100, 100)
            }
        }
    }

    pv = {
        "name": "pv",
        "cls": PVEnv,
        "config": {
            "profile_csv": "pv_profile.csv",
            "scaling_factor": 10.,
            "rescale_spaces": False
        }
    }

    storage = {
        "name": "storage",
        "cls": EnergyStorageEnv,
        "config": {"rescale_spaces": False}
    }

    # Once the individual components are defined, we create a components list that
    # the multi-component agent will use to construct them internally.
    return [building, pv, storage]


@pytest.fixture(scope="function")
def ev_charging_config():
    return {
        "num_vehicles": 100,
        "minutes_per_step": 5,
        "max_charge_rate_kw": 7.,
        "peak_threshold": 250.,
        "vehicle_multiplier": 5.,
        "rescale_spaces": False
    }

@pytest.fixture(scope="function")
def pv_config():
    return {
        "name": "pv",
        "profile_csv": "pv_profile.csv",
        "scaling_factor": 10.
    }


@pytest.fixture(scope="function")
def pv_array_config():
    return {
        "name": "pv",
        "profile_csv": "pv_profile.csv",
        "scaling_factor": 400.
    }


@pytest.fixture
def energy_storage_config():
    return {}
