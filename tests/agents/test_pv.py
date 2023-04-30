from tests.conftest import single_agent_episode_runner

from gridworld.agents.pv import PVEnv


def test_default_env(pv_config):
    """Creates an env instance and runs it (sanity check)."""
    env = PVEnv(**pv_config)
    assert single_agent_episode_runner(env)

