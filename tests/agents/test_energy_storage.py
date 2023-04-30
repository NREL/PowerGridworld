from tests.conftest import single_agent_episode_runner

from gridworld.agents.energy_storage import EnergyStorageEnv


def test_default_env(energy_storage_config):
    """Creates an env instance and runs it (sanity check)."""
    env = EnergyStorageEnv(**energy_storage_config)
    assert single_agent_episode_runner(env)


