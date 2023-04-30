from tests.conftest import single_agent_episode_runner

from gridworld.agents.vehicles import EVChargingEnv

def test_default_env(ev_charging_config):
    """Creates an env instance and runs it (sanity check)."""
    env = EVChargingEnv(**ev_charging_config)
    assert single_agent_episode_runner(env)

