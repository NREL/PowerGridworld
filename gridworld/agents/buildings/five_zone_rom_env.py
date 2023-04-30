from abc import abstractmethod
from collections import OrderedDict
import os
import pickle
import re
from typing import Tuple, Union

import pandas as pd
import numpy as np

import gymnasium as gym

from gridworld.log import logger
from gridworld import ComponentEnv
from gridworld.utils import to_scaled, to_raw, maybe_rescale_box_space
from gridworld.agents.buildings.obs_space import make_obs_space
from gridworld.agents.buildings import defaults
from gridworld.agents.buildings import five_zone_rom_dynamics as dyn


# Below are control variables' boundary.
MAX_FLOW_RATE = [2.2, 2.2, 2.2, 2.2, 3.2]  # Max flow rate for each individual zone
MIN_FLOW_RATE = [.22, .22, .22, .22, .32]  # Max flow rate for each individual zone
MAX_TOTAL_FLOW_RATE = 10.0  # Total flow rate for all zones should be lower than 10 kg/sec.
MAX_DISCHARGE_TEMP = 16.0   # Max temp of air leaving chiller
MIN_DISCHARGE_TEMP = 10.0   # Min temp of air leaving chiller
DEFAULT_COMFORT_BOUNDS = (22., 28.)   # Temps between these values are considered "comfortable"


def load_data(start_time: str = None, end_time: str = None) -> Tuple[pd.DataFrame, dict]:
    """Returns exogenous data dataframe, and state space model (per-zone) dict."""

    THIS_DIR = os.path.dirname(os.path.abspath(__file__))

    df = pd.read_csv(os.path.join(THIS_DIR, "data/exogenous_data.csv"), index_col=0)
    df.index = pd.DatetimeIndex(df.index)

    start_time = pd.Timestamp(start_time) if start_time else df.index[0]
    end_time = pd.Timestamp(end_time) if end_time else df.index[-1]

    _df = df.loc[start_time:end_time]

    if _df is None or len(_df) == 0:
        raise ValueError(
            f"start and/or end times ({start_time}, {end_time}) " +
            "resulted in empty dataframe.  First and last indices are " +
            f"({df.index[0]}, {df.index[-1]}), choose values in this range.")

    with open(os.path.join(THIS_DIR, "data/state_space_model.p"), "rb") as f:
        models = pickle.load(f)

    return _df, models


def get_col(df, pattern, index=None):
    """Returns a dataframe with columns matching regex pattern."""
    return df[[c for c in df.columns if re.match(pattern, c)]].values
    

class FiveZoneROMEnv(ComponentEnv):

    time: pd.Timestamp = None
    time_index: int = None
    raw_action: np.ndarray = None
    state: OrderedDict = None


    def __init__(
        self, 
        name: str = None,
        obs_config: dict = None,
        start_time: Union[str, pd.Timestamp] = None,
        end_time: Union[str, pd.Timestamp] = None,
        comfort_bounds: Union[tuple, np.ndarray, pd.DataFrame] = None,
        zone_temp_init: np.ndarray = None,
        max_episode_steps: int = None,
        rescale_spaces: bool = True,
        **kwargs
    ):

        super().__init__(name=name)

        self.rescale_spaces = rescale_spaces
        self.num_zones = 5
        self.obs_config = obs_config if obs_config is not None else defaults.obs_config

        # Set the initial zone temperature profile.
        if zone_temp_init is not None:
            self.zone_temp_init = zone_temp_init.copy()
        else:
            self.zone_temp_init = 27. * np.ones(self.num_zones, dtype=np.float64)

        # Load exogenous and model data.
        self.df, self.models = load_data(start_time, end_time)

        # Configure max episode steps.
        max_steps = self.df.shape[0] - 3      # due to filter update
        if max_episode_steps is None:
            self.max_episode_steps = max_steps
        else:
            self.max_episode_steps = min(max_episode_steps, max_steps)

        # The default range on comfort bounds are (lowest of low, highest of high)
        self.comfort_bounds = comfort_bounds if comfort_bounds is not None \
            else DEFAULT_COMFORT_BOUNDS

        # Action space:  [zone_flows] + [discharge temp]
        self.act_low = np.array(MIN_FLOW_RATE + [MIN_DISCHARGE_TEMP])
        self.act_high = np.array(MAX_FLOW_RATE + [MAX_DISCHARGE_TEMP])
        self._action_space = gym.spaces.Box(
            low=self.act_low,
            high=self.act_high,
            dtype=np.float64
        )
        self.action_space = maybe_rescale_box_space(
            self._action_space, rescale=self.rescale_spaces)

        # State space is configured via obs_config.
        self.comfort_bounds_df = self.make_comfort_bounds_df()
        self._observation_space, self._obs_labels = make_obs_space(
            self.num_zones, self.obs_config)
        self.observation_space = maybe_rescale_box_space(
            self._observation_space, rescale=self.rescale_spaces)


    def make_comfort_bounds_df(self) -> pd.DataFrame:
        """Returns a dataframe containing upper and lower comfort bounds on the 
        zone temperatures."""
        
        data = np.zeros((self.df.shape[0], 2))
        if isinstance(self.comfort_bounds, tuple):
            data[:, 0], data[:, 1] = self.comfort_bounds[0], self.comfort_bounds[1]
        else:
            data[:, 0] = self.comfort_bounds[:data.shape[0], 0]
            data[:, 1] = self.comfort_bounds[:data.shape[0], 1]

        return pd.DataFrame(data, columns=["temp_lb", "temp_ub"], index=self.df.index)


    def _set_exogenous(self):
        self.temp_oa = get_col(self.df, "T_oa")[self.time_index][0]
        self.q_solar = get_col(self.df, "Q_solar")[self.time_index]
        self.q_cool = get_col(self.df, "Q_cool_", )[self.time_index, :]
        self.q_int = get_col(self.df, "Q_int")[self.time_index]


    def reset(self, **obs_kwargs) -> np.ndarray:
        """Resets the environment to the initial state and returns this state."""

        self.time_index = 0
        self.time = self.df.index[self.time_index]
        self.state = None

        # Set initial state values and exogenous data.
        self.zone_temp = self.zone_temp_init.copy()
        self._set_exogenous()
        self.p_consumed = 0.
        
        # Build the u-vector given current state and exogenous data.
        self.u = dyn.build_u_vector(
            self.models,
            zone_temp=self.zone_temp,
            action=None,
            temp_oa=self.temp_oa,
            q_solar=self.q_solar,
            q_int=self.q_int,
            q_cool=self.q_cool
        )

        # Filter update x2.
        for _ in range(2):
            self.models = dyn.filter_update(
                self.models, self.zone_temp, self.u)

        # Update the zone temperatures based on the filter update.
        self.zone_temp = dyn.temp_dynamics(self.models)

        obs, _ = self.get_obs(**obs_kwargs)

        return obs


    def step(self, action: np.ndarray, **obs_kwargs) -> Tuple[np.ndarray, float, bool, dict]:
        if self.rescale_spaces:
            action = to_raw(action, self._action_space.low, self._action_space.high)
        return self.step_(action, **obs_kwargs)


    def step_(
        self,
        action: np.ndarray,
        **obs_kwargs
    ) -> Tuple[np.ndarray, float, bool, dict]:
        """Applies the action to the system and takes a time step.  Returns
        the new state, stage reward, boolean to indicate whether state is terminal,
        and dictionary of any desired metadata.  In some settings, the p setpoint
        will be updated exogenously."""

        action = np.array(action).squeeze()
        self.raw_action = action

        # Advance the dynamics and update the model and state variables.
        self.model, self.zone_temp = dyn.dynamics(
            self.models,
            self.zone_temp,
            action,
            self.temp_oa,
            self.q_solar,
            self.q_int
        )

        self.p_consumed = dyn.get_p_consumed(action, self.temp_oa)

        # Get the reward
        rew, _ = self.step_reward()

        # Step in time and update the exogenous
        self.time_index += 1
        self.time = self.df.index[self.time_index]
        self._set_exogenous()

        # Call get_obs before returning so state dict is updated.
        obs, state = self.get_obs(**obs_kwargs)

        return np.array(obs), rew, self.is_terminal(), state


    def get_obs(
        self,
        **obs_kwargs
    ) -> Tuple[np.ndarray, dict]:
        """Returns the current state, clipping the values as specified by the 
        gym observation space box constraints.  Calling this method also updates
        the state dict attribute for convenience."""

        # Call the ROM model to get the new zone temps
        
        # Compute the temperature violation per zone
        temp_lb = self.comfort_bounds_df["temp_lb"][self.time].copy()
        temp_ub = self.comfort_bounds_df["temp_ub"][self.time].copy()
        zone_upper_temp_viol = np.zeros(self.num_zones, dtype=np.float64)
        zone_lower_temp_viol = np.zeros(self.num_zones, dtype=np.float64)
        for i, temp in enumerate(self.zone_temp):
            # zone_temp_viol[i] = max(max(0, temp_lb - temp), max(0, temp - temp_ub))
            # Positive violation is true violation while negative violation means margin.
            zone_upper_temp_viol[i] = temp - temp_ub
            zone_lower_temp_viol[i] = temp_lb - temp

        # Add nominal values for bus_voltage and p_setpoint if not provided in kwargs
        bus_voltage = obs_kwargs.get("bus_voltage")
        p_setpoint = obs_kwargs.get("p_setpoint")

        # Create a dict to record all possible state values. We can then filter
        # them out using the obs_config when creating the obs array.
        # TODO: Automate making sure state keys have same order as DEFAULT_OBS_CONFIG.
        self.state = OrderedDict({"zone_temp_{}".format(k): v for k, v in enumerate(self.zone_temp)})
        self.state.update({"zone_upper_viol_{}".format(k): v for k, v in enumerate(zone_upper_temp_viol)})
        self.state.update({"zone_lower_viol_{}".format(k): v for k, v in enumerate(zone_lower_temp_viol)})
        self.state.update({
            "comfort_lower": temp_lb,       # current comfort lower bound
            "comfort_upper": temp_ub,       # current comfort upper bound
            "outdoor_temp": self.temp_oa,   # current outdoor temp
            "p_consumed": self.p_consumed,  # current p consumed 
            "time_of_day": 1. * self.time_index / self.max_episode_steps, # time,
            "bus_voltage": bus_voltage if bus_voltage is not None else 1.0,
            "min_voltage": bus_voltage if bus_voltage is not None else 1.0,
            "max_voltage": bus_voltage if bus_voltage is not None else 1.0,
            "p_setpoint": p_setpoint if p_setpoint is not None else np.inf
        })
        self.state.update(obs_kwargs)

        # Create the filtered observation array and clip values to low/high
        obs = np.array(
            [v for k, v in self.state.items() if k in self.obs_labels],
            dtype=object   # otherwise a warning is raised about ragged seq
        ).astype(np.float64)

        obs = np.clip(obs, self._observation_space.low, self._observation_space.high).squeeze()
        
        if self.rescale_spaces:
            obs = to_scaled(obs, self._observation_space.low, self._observation_space.high)

        return obs.copy(), self.state.copy()


    def step_reward(self) -> Tuple[float, dict]:
        """Default reward is soft constraint on comfort bounds."""

        viol_lower = [v for k,v in self.state.items() if k.startswith("zone_upper_viol_")]
        viol_upper = [v for k,v in self.state.items() if k.startswith("zone_upper_viol_")]
        
        rew = np.array(viol_lower)**2 + np.array(viol_upper)**2
        
        return rew, {}


    def is_terminal(self) -> bool:
        """Returns whether the current state is terminal.  Currently this is only
        true when the maximum number of episode steps is reached."""

        return self.time_index == self.max_episode_steps - 1


    @property
    def real_power(self) -> float:
        """Return the real power consumed in the most recent step."""

        return self.state["p_consumed"]



class FiveZoneROMThermalEnergyEnv(FiveZoneROMEnv):
    """Subclass with identical physics, but that balances energy and comfort costs."""

    def step_reward(self) -> Tuple[float, dict]:
        """Overwriting reward to balance energy and comfort."""

        alpha = 0.2

        energy_consumption_reward = -self.state["p_consumed"] / 12.0

        comfort_error = [
            max(self.state["zone_upper_viol_{}".format(i)], self.state["zone_lower_viol_{}".format(i)], 0.0)
            for i in range(self.num_zones)
        ]
        comfort_reward = -(sum([x**2 for x in comfort_error]))

        reward = alpha * energy_consumption_reward * 0.5 + (1. - alpha) * comfort_reward

        meta = {
            "comfort_rew": comfort_reward,
            "energy_rew": energy_consumption_reward
        }

        return reward, meta


if __name__ == '__main__':

    env = FiveZoneROMThermalEnergyEnv()

    obs = env.reset()

    print(obs)