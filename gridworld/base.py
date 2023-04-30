from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Tuple, List, Dict

import numpy as np

import gymnasium as gym

from gridworld.log import logger


class ComponentEnv(gym.Env, ABC):
    """Base class for any environment used in the multiagent simulation."""

    def __init__(
        self,
        name: str = None,
        **kwargs
    ):
        super().__init__()
        
        self.name = name
        self._real_power = 0.
        self._reactive_power = 0.
        self._obs_labels = []


    @abstractmethod
    def reset(self, **kwargs) -> Tuple[np.ndarray, dict]:
        """Standard gym reset method but with kwargs."""
        return


    @abstractmethod
    def step(self, action: np.ndarray, **kwargs) -> Tuple[np.ndarray, float, bool, dict]:
        """Standard gym step method but with kwargs."""
        return


    @abstractmethod
    def step_reward(self, **kwargs) -> Tuple[float, dict]:
        """Returns the current step reward and metadata dict."""
        return
    

    @abstractmethod
    def get_obs(self, **kwargs) -> Tuple[np.ndarray, dict]:
        """Returns the current observation (state) and any metadata."""
        return


    @property
    def real_power(self) -> float:
        """Returns the real power of the component, positive for load and 
        negative for generation."""
        return self._real_power


    @property
    def reactive_power(self) -> float:
        """Returns the reactive power of the component, positive for load and 
        negative for generation."""
        return self._reactive_power


    @property
    def obs_labels(self) -> list:
        """Returns a list of observation variable labels.  External variables
        coming from the multiagent env must be included here to indicate that
        they are needed by the env.  Otherwise this is optional."""
        return self._obs_labels


class MultiComponentEnv(ComponentEnv):
    """Class for creating a single Gym environment from multiple component 
    environments.  The action and observation spaces of the multi-component env
    are taken as the union over the components.
    """

    def __init__(
            self, 
            name: str = None,
            components: List[dict] = None,
            **kwargs
        ):

        super().__init__(name=name, **kwargs)
        
        self.envs = []
        for c in components:
            env = c["cls"](name=c["name"], **c["config"])
            self.envs.append(deepcopy(env))

        self.observation_space = gym.spaces.Dict(
            {e.name: e.observation_space for e in self.envs})

        self.action_space = gym.spaces.Dict(
            {e.name: e.action_space for e in self.envs})

        self._obs_labels_dict = {e.name: e.obs_labels for e in self.envs}
        obs_labels = []
        for e in self.envs:
            obs_labels += e.obs_labels
        self._obs_labels = list(set(obs_labels))


    def reset(self, **kwargs) -> dict:
        """Default reset method resets each component and returns the obs dict."""
        _ = [e.reset(**kwargs) for e in self.envs]
        return self.get_obs(**kwargs)


    def step(self, action: dict, **kwargs) -> Tuple[dict, float, bool, dict]:
        """Default step method composes the obs, reward, done, meta dictionaries
        from each component step."""

        # Initialize outputs.
        real_power = 0.
        obs = {}
        dones = []
        metas = {}

        # Loop over envs and collect real power injection/consumption.
        for env in self.envs:
            env_kwargs = {k: v for k,v in kwargs.items() if k in env.obs_labels}
            ob, _, done, meta = env.step(action[env.name], **env_kwargs)
            obs[env.name] = ob.copy()
            dones.append(done)
            metas[env.name] = meta.copy()
            real_power += env.real_power

        # Set real power attribute.  TODO:  Reactive power.
        self._real_power = real_power

        # Compute the step reward using user-implemented method.
        step_reward, _ = self.step_reward()
        
        return obs, step_reward, any(dones), metas


    def step_reward(self) -> Tuple[float, dict]:
        """Default step reward simply sums those from the components.  Overwrite
        this method to customize how this is computed."""

        # Initialize outputs.
        reward = 0.
        meta = {}

        # Loop over envs and create the reward dict.
        for env in self.envs:
            r, m = env.step_reward()
            reward += r
            meta[env.name] = m.copy()

        return reward, meta


    def get_obs(self, **kwargs) -> Tuple[dict, dict]:
        """Default get obs composes a dictionary of observations from each 
        component env."""

        # Initialize outputs.
        obs = {}
        meta = {}

        # Loop over envs and create the observation dict (of dicts).
        for env in self.envs:
            env_kwargs = {k: v for k,v in kwargs.items() if k in env.obs_labels}
            obs[env.name], meta[env.name] = env.get_obs(**env_kwargs)

        return obs, meta


    @property
    def obs_labels_dict(self) -> Dict[str, list]:
        return self._obs_labels_dict


    @property
    def env_dict(self) -> Dict[str, ComponentEnv]:
        return {e.name: e for e in self.envs}
