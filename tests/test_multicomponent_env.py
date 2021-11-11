from tests.conftest import multicomponent_building_config, multi_agent_episode_runner

from gridworld import MultiComponentEnv


def test_default_multicomponent_env(multicomponent_building_config):

    env = MultiComponentEnv(components=multicomponent_building_config)

    return multi_agent_episode_runner(env)


