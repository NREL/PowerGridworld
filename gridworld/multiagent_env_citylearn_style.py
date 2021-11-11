
from collections import OrderedDict

import gym
import numpy as np

from gridworld.multiagent_env import MultiAgentEnv

# TODO:  Can this share a common base class with the OpenAI version?

class MultiagentEnvCityLearnStyle(gym.Env):
    """ A wrapper class to convert RLLib multi-agent gym env to the CityLearn style.

    """

    def __init__(self, env_config):

        self.ma_env = MultiAgentEnv(**env_config)

        # nested_sequence is used as reference to keep the sequence correct when 
        # putting data into CityLearn arrays.
        self.nested_sequence = self.get_nested_sequence(env_config['agents'])

        # Definitions below for observation_space(s) and action_space(s) follow 
        # the CityLearn example.
        self.observation_spaces = []
        self.action_spaces = []
        total_agents_obs_len = 0
        total_agents_act_len = 0

        for k, v in self.nested_sequence.items():

            agent_obs_len = sum(
                [self.ma_env.observation_space[k][component].shape[0] for component in v])
            agent_act_len = sum(
                [self.ma_env.action_space[k][component].shape[0] for component in v])

            agent_obs_space = gym.spaces.Box(
                shape=(agent_obs_len,),
                low=-1.0,
                high=1.0,
                dtype=np.float64
            )
            
            agent_action_space = gym.spaces.Box(
                shape=(agent_act_len,),
                low=-1.0,
                high=1.0,
                dtype=np.float64
            )
            
            self.observation_spaces.append(agent_obs_space)
            self.action_spaces.append(agent_action_space)

            total_agents_act_len += agent_act_len
            total_agents_obs_len += agent_obs_len

        self.observation_space = gym.spaces.Box(
            shape=(total_agents_obs_len,),
            low=-1.0,
            high=1.0,
            dtype=np.float64
        )
        
        self.action_space = gym.spaces.Box(
            shape=(total_agents_act_len,),
            low=-1.0,
            high=1.0,
            dtype=np.float64
        )


    @staticmethod
    def get_nested_sequence(config):
        results = OrderedDict()
        for item in config:
            results[item['name']] = [x['name'] for x in item['config']['components']]
        return results


    def reset(self):

        obs = self.ma_env.reset()
        obs_citylearn_style = self.convert_to_citylearn_obs(obs)

        return obs_citylearn_style


    def step(self, action):

        action = self.convert_from_citylearn_act(action)

        next_obs, reward, done, info = self.ma_env.step(action)

        next_obs_citylearn_style = self.convert_to_citylearn_obs(next_obs)
        reward_citylearn_style = [reward[k] for k in self.nested_sequence.keys()]
        done = done['__all__']

        return next_obs_citylearn_style, reward_citylearn_style, done, info


    def convert_to_citylearn_obs(self, obs):
        """Convert the RLLib dictionary based observation to CityLearn array 
        based observation.
        """

        obs_cl = []
        for k, v in self.nested_sequence.items():
            obs_cl.append(np.concatenate([obs[k][x] for x in v]))

        return np.array(obs_cl)


    def convert_from_citylearn_act(self, action):
        """Convert the CityLearn array based action into RLLib dictionary based action.
        """

        converted_action = {}
        idx = 0

        for k, v in self.nested_sequence.items():
            agent_action = OrderedDict()
            act_start_idx = 0
            for component in v:
                act_len = self.ma_env.action_space[k][component].shape[0]
                agent_action[component] = action[idx][act_start_idx: act_start_idx + act_len]
                act_start_idx += act_len

            converted_action[k] = agent_action
            idx += 1

        return converted_action


if __name__ == '__main__':

    from gridworld.scenarios.ieee_13_bus_buildings import make_env_config

    env_config = make_env_config(
        building_config={
            "reward_structure": {"target": "min_voltage", "alpha": 0.5}
        },
        pv_config={
            "profile_csv": "off-peak.csv",
            "scaling_factor": 40.
        },
        storage_config={
            "max_power": 20.,
            "storage_range": (3., 250.)
        },
        system_load_rescale_factor=0.6,
        num_buildings=3
    )

    env = MultiagentEnvCityLearnStyle(env_config)

    print("******** Test 1 **********")
    obs_rllib = env.ma_env.reset()
    obs_citylearn = env.convert_to_citylearn_obs(obs_rllib)

    print(obs_rllib)
    print(obs_citylearn)

    print()
    print("******** Test 2 **********")
    acts_citylearn = [x.sample() for x in env.action_spaces]
    acts_rllib = env.convert_from_citylearn_act(acts_citylearn)

    print(acts_rllib)
    print(acts_citylearn)

    print()
    print("******** Test 3 **********")

    print(env.reset())
    done = False

    while not done:
        acts = [x.sample() for x in env.action_spaces]
        new_state, reward, done, info = env.step(acts)

