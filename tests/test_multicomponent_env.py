from tests.conftest import multi_agent_episode_runner

from gridworld import MultiComponentEnv


def test_default_multicomponent_env(multicomponent_building_config):

    env = MultiComponentEnv(components=multicomponent_building_config)

    assert multi_agent_episode_runner(env)
