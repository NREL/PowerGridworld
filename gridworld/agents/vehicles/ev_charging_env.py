from collections import OrderedDict
import os
from typing import Tuple

import numpy as np
import pandas as pd

import gym

from gridworld.log import logger
from gridworld import ComponentEnv
from gridworld.utils import maybe_rescale_box_space, to_raw, to_scaled

THIS_DIR = os.path.dirname(os.path.abspath(__file__))


class EVChargingEnv(ComponentEnv):

    def __init__(
        self,
        num_vehicles: int = 100,
        minutes_per_step: int = 5,
        max_charge_rate_kw: float = 7.0,  # ~40. for fast charge
        max_episode_steps: int = None,
        unserved_penalty: float = 1.,
        peak_penalty: float = 1.,
        peak_threshold: float = 10.,
        reward_scale: float = 1e5,
        name: str = None,
        randomize: bool = False,
        vehicle_csv: str = None,
        vehicle_multiplier: int = 1,
        rescale_spaces: bool = True,
        **kwargs
    ):

        super().__init__(name=name)

        self.num_vehicles = num_vehicles
        self.max_charge_rate_kw = max_charge_rate_kw
        self.minutes_per_step = minutes_per_step
        self.randomize = randomize
        self.vehicle_multiplier = vehicle_multiplier
        self.rescale_spaces = rescale_spaces

        # Reward parameters
        self.unserved_penalty = unserved_penalty
        self.peak_penalty = peak_penalty
        self.peak_threshold = peak_threshold
        self.reward_scale = reward_scale

        # By default, we simulate a whole day but allow user to specify
        # fewer steps if desired.
        self.max_episode_steps = max_episode_steps if max_episode_steps is not None else np.inf
        self.max_episode_steps = min(self.max_episode_steps, 24*60 / minutes_per_step)

        # Create an array of simulation times in minutes, in the interval
        # (0, max_episode_steps * minutes_per_step).
        self.simulation_times = np.arange(
            0, self.max_episode_steps * minutes_per_step, minutes_per_step)

        # Attributes that will be initialized in reset.
        self.time_index = None  # time index
        self.time = None  # time in minutes
        self.df = None    # episode vehicle dataframe
        self.charging_vehicles = None  # charging vehicle list
        self.departed_vehicles = None  # vehicle list departed in last time step

        # Read the source dataframe.
        vehicle_csv = vehicle_csv if vehicle_csv else os.path.join(THIS_DIR, "vehicles.csv")
        self._df = pd.read_csv(vehicle_csv)     # all vehicles
        self._df["energy_required_kwh"] *= self.vehicle_multiplier

        # Round the start/end times to the nearest step.
        self._df["start_time_min"] = self._round(self._df["start_time_min"])
        self._df["end_time_park_min"] = self._round(self._df["end_time_park_min"])

        # Bounds on the observation space variables.
        obs_bounds = OrderedDict({
            "time": (0, self.simulation_times[-1]),
            "num_active_vehicles": (
                0, self.num_vehicles),
            "real_power_consumed": (
                0, self.num_vehicles * self.max_charge_rate_kw),
            "real_power_demand": (
                0, self.num_vehicles * self._df["energy_required_kwh"].max()),
            "mean_charge_rate_deficit": (
                0, self._df["energy_required_kwh"].max() / (self.minutes_per_step / 60.)),
            "real_power_unserved": (
                0, self._df["energy_required_kwh"].max())
        })

        # Construct the gym spaces.
        self._observation_space = gym.spaces.Box(
            low=np.array([x[0] for x in obs_bounds.values()]),
            high=np.array([x[1] for x in obs_bounds.values()]),
            shape=(len(obs_bounds), ),
            dtype=np.float64)
        self.observation_space = maybe_rescale_box_space(
            self._observation_space, rescale=self.rescale_spaces)

        # Fraction between 0 and 1 of max charge rate for all charging vehicles.
        self._action_space = gym.spaces.Box(
            low=0.,
            high=1.,
            shape=(1, ),
            dtype=np.float64
        )
        self.action_space = maybe_rescale_box_space(
            self._action_space, rescale=self.rescale_spaces)

        # Use a dictionary to keep track of various state quantities.
        # Use the self._update(key, value) to ensure valid keys when updating state.
        self.state = OrderedDict({k: None for k in obs_bounds.keys()})

        # Use the state dict to create the observation labels.
        self._obs_labels = list(self.state.keys())


    def get_obs(self, **kwargs) -> Tuple[dict, dict]:
        "Returns an observation dict and metadata dict."
        raw_obs = np.array(list(self.state.values()))
        if self.rescale_spaces:
            obs = to_scaled(raw_obs, self._observation_space.low, self._observation_space.high)
        else:
            obs = raw_obs
        return obs.copy(), self.state.copy()


    def is_terminal(self) -> bool:
        """Returns True if max episode steps have been reached."""
        return self.time_index == self.max_episode_steps - 1


    def step_reward(self) -> Tuple[float, dict]:
        """Return a non-zero reward here if you want to use RL."""
        unserved_reward = -self.unserved_penalty * self.state["real_power_unserved"]**2
        peak_reward = -self.peak_penalty * \
            max(0, self.state["real_power_consumed"] - self.peak_threshold)**2
        reward = unserved_reward + peak_reward
        reward /= self.reward_scale
        return reward, {"real_power_unserved": unserved_reward, "peak_reward": peak_reward}


    def reset(self, **kwargs) -> Tuple[dict, dict]:
        """Reset the initial conditions and run a single step of the simulation
        so that `get_obs` here can be used in the first control step."""

        self.time_index = 0 
        self.time = self.simulation_times[self.time_index]
        self.charging_vehicles = []
        self.departed_vehicles = []

        # Select first N vehicles if not randomized, else shuffle rows of df.
        self.df = self._df.sample(self.num_vehicles).copy() if self.randomize \
            else self._df[:self.num_vehicles].copy()
        self.df = self.df.reset_index()     # index is now 0 to N-1

        # Initialize real power.
        self._real_power = 0.
        
        # Step the simulator one time without a control action.
        self.step()

        # Get the observation needed to solve the first control step.
        obs, _ = self.get_obs()
        
        return obs, {}


    def step(self, action: np.ndarray = None, **kwargs) -> Tuple[np.ndarray, float, bool, dict]:

        logger.debug(f'Time index {self.time_index}/{self.max_episode_steps}')
        logger.debug(f'Action: {action}')

        # If no action is applied, use minimum.
        # TODO: Make sure you are scaling things correctly.
        action = action if action is not None else self._action_space.low
        if self.rescale_spaces:
            action = to_raw(action, self._action_space.low, self._action_space.high)

        action_kw = action[0] * self.max_charge_rate_kw
        action_kwh = action_kw * (self.minutes_per_step / 60.)

        # Get indexes of vehicles arriving and departing.
        start_idx = np.where(self.time >= np.floor(self.df["start_time_min"]))[0]
        end_idx = np.where(self.time <= np.floor(self.df["end_time_park_min"]))[0]

        # Get indexes of charging vehicles.
        charging_vehicles = list(set(list(start_idx)).intersection(set(list(end_idx))))
        charging_vehicles = [i for i in charging_vehicles if self.df.at[i, "energy_required_kwh"] > 0.]

        # Get vehicles that have left the station in the last time step.
        self.departed_vehicles = list(set(self.charging_vehicles) - set(charging_vehicles))

        logger.debug(f"STEP, {self.time}, {self.time_index}, {charging_vehicles}, {self.departed_vehicles}")

        # Aggregate quantities that are needed for obs space.
        real_power_consumed = 0.
        real_power_demand = 0.
        min_energy_required = 0.
        charge_rate_deficit = []  # charge rate missing to reach full charge

        for i in charging_vehicles:

            # Compute energy required to fully charge.
            energy_required_kwh = self.df["energy_required_kwh"][i]

            # Update the aggregate variables
            real_power_demand += energy_required_kwh
            min_energy_required = max(min_energy_required, energy_required_kwh)

            # If the vehicle does not require any more charging then skip it.
            if energy_required_kwh <= 0.:
                continue

            # What is the min energy this vehicle needs to reach full charge?
            time_left_h = (self.df["end_time_park_min"][i] - self.time) / 60.
            if time_left_h <= 0:
                continue
            deficit = max(
                0, self.max_charge_rate_kw - energy_required_kwh / time_left_h)
            charge_rate_deficit.append(deficit)

            # Apply action and update the vehicle data.
            charge_energy_kwh = min(action_kwh, energy_required_kwh)
            self.df.at[i, "energy_required_kwh"] -= charge_energy_kwh
            real_power_consumed += charge_energy_kwh

            # print(action_kwh, energy_required_kwh, real_power_consumed)

            logger.debug(f"{i}, {energy_required_kwh}, {action}")
            
        # Update time variables.
        self.time_index += 1
        self.time = self.simulation_times[self.time_index]
        self.charging_vehicles = charging_vehicles

        # Compute unmet charging demand for departed vehicles.
        unserved = 0.
        for i in self.departed_vehicles:
            unserved  += self.df["energy_required_kwh"][i]
        self._update("real_power_unserved", unserved)

        # Update the state dict.
        self._update("time", self.time)
        self._update("num_active_vehicles", self.vehicle_multiplier * len(charging_vehicles))
        self._update("real_power_consumed", self.vehicle_multiplier * real_power_consumed)
        self._update("real_power_demand", self.vehicle_multiplier * real_power_demand)
        self._update(
            "mean_charge_rate_deficit",
            0 if len(charge_rate_deficit) == 0 else np.mean(charge_rate_deficit))

        # Update the real power attribute needed for component envs.
        self._real_power = self.vehicle_multiplier * real_power_consumed
       
        # Get the return values
        obs, meta = self.get_obs()
        rew, rew_meta = self.step_reward()
        done = self.is_terminal()

        meta.update(rew_meta)

        return obs, rew, done, meta

    
    def _update(self, key, value):
        if key not in self.state:
            raise ValueError(f'Invalid state key {key}')
        self.state[key] = value


    def _round(self, x):
        """Round the value x down to the nearest time step interval."""
        return x - x % self.minutes_per_step

