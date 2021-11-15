from abc import abstractmethod
from collections import defaultdict

import logging
from typing import Dict, Tuple, Union

import numpy as np
import pandas as pd

from gridworld.base import ComponentEnv, MultiComponentEnv
from gridworld.log import logger

try:
    from ray.rllib.env.multi_agent_env import MultiAgentEnv as Env
except ImportError:
    Env = object
    logger.warning("rllib MultiAgentEnv not found, using generic object class")


class MultiAgentEnv(Env):
    """This class implements the multi-agent environment created from a list
    of agents of either type, ComponentEnv or MultiComponentEnv."""

    def __init__(
            self,
            common_config: dict = {},
            pf_config: dict = {},
            agents: list = None,
            max_episode_steps: int = None,
            rescale_spaces: bool = True,
            **kwargs
    ):

        self.common_config = common_config
        self.rescale_spaces = rescale_spaces
        assert len(agents) > 0, "need at least one agent!"

        # TODO:  If we required certain keys in this config dict, we need
        # to do some simple checking and raise a helpful error.
        self.start_time = pd.Timestamp(common_config["start_time"])
        self.end_time = pd.Timestamp(common_config["end_time"])
        self.control_timedelta = common_config["control_timedelta"]

        # Likewise here, need helpful checking and errors.
        self.pf_config = pf_config
        self.max_episode_steps = max_episode_steps if max_episode_steps is not None else np.inf
        
        self.episode_step = None
        self.time = None
        self.history = None
        self.voltages = None
        self.obs_dict = {}

        # Call the agent constructors with both common and agent-specific 
        # configuration arguments.
        self.agents = []
        for a in agents:

            # Top-level name argument overrides a name in the config.  The
            # constructor will error out otherwise because it gets two values
            # of the argument, one from config dict and one from the agent name.
            _config = a["config"]
            if "name" in a["config"]:
                _config = {k: v for k, v in _config.items() if k != "name"}
                logger.warning(
                    f"ignoring 'name' in config dict in favor of constructor argument")
            
            # Call the constructor and append to the agent list.
            new_agent = a["cls"](name=a["name"], **_config, **self.common_config)
            self.agents.append(new_agent)

        # Keep track of which bus each agent is attached to.    
        self.agent_name_bus_map = {a["name"]: a["bus"] for a in agents} 
        
        # Create a list of agent names and ensure they are unique.
        self.agent_names = list(set([a.name for a in self.agents]))
        assert len(self.agent_names) == len(agents), "all agents need unique names"

        # Instantiate the powerflow solver.
        self.pf_solver = pf_config["cls"](**pf_config["config"])

        # Create the gym observation and action spaces.
        self.observation_space = {
            agent.name: agent.observation_space for agent in self.agents}
        self.action_space = {
            agent.name: agent.action_space for agent in self.agents}


    @abstractmethod
    def get_external_obs_vars(
        self, 
        agent: Union[ComponentEnv, MultiComponentEnv]
    ) -> dict:
        """These are external variables to the agents, need to implement how
        they get this data so it can be passed to their reset/step methods
        and added to the observation space.  Currently, a user will have to
        overwrite the method to give agents access to other quantities.
        TODO: Design an interface for a user to customize this."""

        kwargs = {}

        # Get the bus voltage at the agent's bus.
        if "bus_voltage" in agent.obs_labels:
            kwargs["bus_voltage"] = self.pf_solver.get_bus_voltage_by_name(
                self.agent_name_bus_map[agent.name])

        # Get the maximum voltage across all buses.
        if "max_voltage" in agent.obs_labels:
            kwargs["max_voltage"] = max(list(self.voltages.values()))

        # Get the minimum voltage across all buses.
        if "min_voltage" in agent.obs_labels:
            kwargs["min_voltage"] = min(list(self.voltages.values()))

        return kwargs

    
    @abstractmethod
    def reward_transform(self, agent_rewards: dict) -> dict:
        """Function to transform the agent rewards based on centralized view.
        Pass-through by default but can be overwrittent for custom rewards."""
        return agent_rewards


    def reset(self) -> Dict[str, any]:

        self.episode_step = 0
        self.time = self.start_time
        self.history = {"timestamp": [], "voltage": [], "agent_power_p": []}

        # Run OpenDSS to have voltage info
        self.pf_solver.calculate_power_flow(current_time=self.time)
        self.voltages = self.pf_solver.get_bus_voltages()

        # Reset the controllable agents and collect their obs arrays
        for agent in self.agents:
            kwargs = self.get_external_obs_vars(agent)
            _ = agent.reset(**kwargs)

        return self.get_obs()


    def get_obs(self) -> Dict[str, any]:
        obs = {}
        for agent in self.agents:
            kwargs = self.get_external_obs_vars(agent)
            obs[agent.name], _ = agent.get_obs(**kwargs)
        return obs


    def step(self, action: Dict[str, any]) -> Tuple[dict, dict, dict, dict]:

        self.episode_step += 1
        self.time += self.control_timedelta
        self.obs_dict = {}

        # Initialize agent outputs.
        obs, rew, done, meta = {}, {}, {}, {}
        load_p, load_q = {}, {}
        agent_power_p = []

        # For each agent, call the step method and inject any external variables
        # as keyword arguments.  Accumulate the real/reactive power from each
        # agent for use in power flow calculation.
        for agent in self.agents:
            name = agent.name
            kwargs = self.get_external_obs_vars(agent)
            obs[name], rew[name], done[name], meta[name] = agent.step(
                action=action[name], **kwargs)

            load_bus = self.agent_name_bus_map[name]
            agent_p_consumed = agent.real_power
            agent_q_consumed = agent.reactive_power
            agent_power_p.append(agent_p_consumed)

            if load_bus in load_p.keys():
                load_p[load_bus] += agent_p_consumed
                load_q[load_bus] += agent_q_consumed
            else:
                load_p[load_bus] = agent_p_consumed
                load_q[load_bus] = agent_q_consumed

        # Call power flow solver and update the bus voltages.
        self.pf_solver.calculate_power_flow(
            current_time=self.time,
            p_controllable_consumed=load_p,
            q_controllable_consumed=load_q
        )
        self.voltages = self.pf_solver.get_bus_voltages()

        # Update history dict.
        self.history["timestamp"].append(self.time)
        self.history["voltage"].append(self.voltages.copy())
        self.history["agent_power_p"].append(agent_power_p)

        # Check for terminal condition.  Currently, we stop the entire simulation
        # if any agent is `done`, although the RLLib API allows agents to finish
        # at different times.
        any_done = np.any(list(done.values()))  # will fail otherwise
        max_steps_reached = (self.episode_step == self.max_episode_steps - 1)
        time_up = self.time >= self.end_time
        done = any_done or max_steps_reached or time_up

        # Create the dones dict that will be returned.  We assume all are done
        # or not simulataneously.
        dones = {a.name: done for a in self.agents}
        dones["__all__"] = done

        rew = self.reward_transform(rew)
        meta = self.meta_transform(meta)

        return obs, rew, dones, meta


    def reward_transform(self, rew_dict) -> dict:
        """Function to transform the agent rewards based on centralized view.
        Pass-through by default."""
        return rew_dict


    def meta_transform(self, meta) -> dict:
        """Function to augment the agent meta info based on centralized view.
        Pass-through by default.
        """
        return meta


    @property
    def agent_dict(self) -> Dict[str, ComponentEnv]:
        return {a.name: a for a in self.agents}
