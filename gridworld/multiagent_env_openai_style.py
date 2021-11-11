
from collections import OrderedDict

import numpy as np

import gym


# TODO:  Can this share a common base class with the CityLearn version?


class MultiagentEnvOpenAIStyle(gym.Env):
    """ A wrapper class to convert RLLib multi-agent gym env to the OpenAI 
    MADDPG style.
    """

    def __init__(self, multi_agent_env_cls, env_config):

        self.ma_env = multi_agent_env_cls(**env_config)
        self.n = len(self.ma_env.agents)

        # nested_sequence is used as reference to keep the sequence correct
        # when putting data into CityLearn arrays.
        self.nested_sequence = self.get_nested_sequence(env_config['agents'])

        # Definitions below for observation_space(s) and action_space(s) follow 
        # the OpenAI example.
        self.observation_space = []
        self.action_space = []
        total_agents_obs_len = 0
        total_agents_act_len = 0

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

            total_agents_act_len += agent_act_len
            total_agents_obs_len += agent_obs_len

    @staticmethod
    def get_nested_sequence(config):
        results = OrderedDict()
        for item in config:
            results[item['name']] = [x['name']
                                     for x in item['config']['components']]
        return results

    def reset(self):

        obs = self.ma_env.reset()
        obs_openai_style = self.convert_to_openai_obs(obs)

        return obs_openai_style

    def step(self, action):

        action = self.convert_from_openai_act(action)

        next_obs, reward, done, info = self.ma_env.step(action)

        next_obs_openai_style = self.convert_to_openai_obs(next_obs)
        reward_openai_style = [reward[k] for k in self.nested_sequence.keys()]
        done = [done[k] for k in self.nested_sequence.keys()]

        return next_obs_openai_style, reward_openai_style, done, info

    def convert_to_openai_obs(self, obs):
        """ Convert the RLLib dictionary based observation to OpenAI array 
        based observation.
        """

        obs_oa = []
        for k, v in self.nested_sequence.items():
            obs_oa.append(np.concatenate([obs[k][x] for x in v]))

        return obs_oa

    def convert_from_openai_act(self, action):
        """ Convert the OpenAI array based action into RLLib dictionary based
            action.
        """

        converted_action = {}
        idx = 0

        for k, v in self.nested_sequence.items():
            agent_action = OrderedDict()
            act_start_idx = 0
            for component in v:
                act_len = self.ma_env.action_space[k][component].shape[0]
                agent_action[component] = action[idx][act_start_idx:
                                                      act_start_idx + act_len]
                act_start_idx += act_len

            converted_action[k] = agent_action
            idx += 1

        return converted_action


if __name__ == '__main__':

    import pprint

    from gridworld.multiagent_env import MultiAgentEnv
    from gridworld.scenarios.ieee_13_bus_buildings import make_env_config

    pp = pprint.PrettyPrinter(indent=2)

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

    env = MultiagentEnvOpenAIStyle(MultiAgentEnv, env_config)

    print("******** Test 1 **********")
    obs_rllib = env.ma_env.reset()
    obs_openai = env.convert_to_openai_obs(obs_rllib)

    pp.pprint(obs_rllib)
    pp.pprint(obs_openai)

    print()
    print("******** Test 2 **********")
    acts_openai = [x.sample() for x in env.action_space]
    acts_rllib = env.convert_from_openai_act(acts_openai)

    pp.pprint(acts_rllib)
    pp.pprint(acts_openai)

    print()
    print("******** Test 3 **********")

    pp.pprint(env.reset())
    done_all = False
    cnt = 0

    while not done_all:
        acts = [x.sample() for x in env.action_space]
        new_state, reward, done, info = env.step(acts)
        done_all = all(done)
        cnt += 1

    print(cnt)