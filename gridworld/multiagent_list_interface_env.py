
from collections import OrderedDict

import gymnasium as gym
import numpy as np


class MultiAgentListInterfaceEnv(gym.Env):
    """ A wrapper class to convert the env's dict interface to list interface.

    By default, the MultiAgentEnv provides interface to RL algorithms using
    a dictionary, e.g., observation_space = {'agent_1': Box(...), 'agent_2':
    Box(...), ...}, and RL training frameworks like RLLib can handle this.
    Other frameworks, however, require the interface to be a list, e.g.,
    action_space = [Box(...), Box(...), ...]. This wrapper class is to convert
    the default dict interface to list interface.

    """

    def __init__(self, multi_agent_env_cls, env_config):

        self.ma_env = multi_agent_env_cls(**env_config)
        self.n = len(self.ma_env.agents)

        # nested_sequence is used as reference to keep the sequence correct
        # when putting data into the list interface format.
        self.nested_sequence = self.get_nested_sequence(env_config['agents'])

        self.observation_space = []
        self.action_space = []

        for k, v in self.nested_sequence.items():

            agent_obs_len = sum(
                [self.ma_env.observation_space[k][component].shape[0]
                 for component in v])
            agent_act_len = sum(
                [self.ma_env.action_space[k][component].shape[0]
                 for component in v])

            agent_obs_space = gym.spaces.Box(shape=(agent_obs_len,),
                                             low=-1.0,
                                             high=1.0,
                                             dtype=np.float64)
            agent_action_space = gym.spaces.Box(shape=(agent_act_len,),
                                                low=-1.0,
                                                high=1.0,
                                                dtype=np.float64)
            self.observation_space.append(agent_obs_space)
            self.action_space.append(agent_action_space)

    @staticmethod
    def get_nested_sequence(agent_config):
        nested_agents_components_sequence = OrderedDict()
        for item in agent_config:
            nested_agents_components_sequence[item['name']] = [x['name']
                                     for x in item['config']['components']]
        return nested_agents_components_sequence

    def reset(self):

        obs = self.ma_env.reset()
        obs_list_interface = self.convert_to_list_obs(obs)

        return obs_list_interface

    def step(self, action):

        action = self.convert_from_list_act(action)

        next_obs, reward, done, info = self.ma_env.step(action)

        next_obs_list_interface = self.convert_to_list_obs(next_obs)
        reward_list_interface = [reward[k]
                                 for k in self.nested_sequence.keys()]
        done = [done[k] for k in self.nested_sequence.keys()]

        return next_obs_list_interface, reward_list_interface, done, info

    def convert_to_list_obs(self, obs):
        """ Convert the dictionary based observation to list based observation.
        """

        obs_list = []
        for k, v in self.nested_sequence.items():
            obs_list.append(np.concatenate([obs[k][x] for x in v]))

        return obs_list

    def convert_from_list_act(self, action):
        """ Convert the list based action into dictionary based action.
        """

        converted_action = {}
        idx = 0

        for k, v in self.nested_sequence.items():

            agent_action = {}
            act_start_idx = 0

            for component in v:
                act_len = self.ma_env.action_space[k][component].shape[0]
                agent_action[component] = action[idx][act_start_idx:
                                                      act_start_idx + act_len]
                act_start_idx += act_len

            converted_action[k] = agent_action
            idx += 1

        return converted_action
