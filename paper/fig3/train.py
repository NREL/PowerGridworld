import os

import ray
from ray import tune
from ray.tune.registry import register_env

from gridworld.scenarios.heterogeneous import make_env_config


def env_creator(config: dict):
    """Simple wrapper that takes a config dict and returns an env instance."""
    
    from gridworld import MultiAgentEnv

    return MultiAgentEnv(**config)


def main(**args):

    # Start ray locally if no redis password provided, else assume we're running
    # on a cluster.
    if args["redis_password"] is None:
        ray.init()
    else:
        ray.init(
            _redis_password=args["redis_password"],
            address=os.environ["ip_head"]    # exported in bash script
        )

    # Register the environment.
    env_name = args["env_name"]
    register_env(env_name, env_creator)

    # Create the env configuration with option to change max episode steps
    # for debugging.
    env_config = make_env_config(
        system_load_rescale_factor=args["system_load_rescale_factor"],
        rescale_spaces=True
    )
    env_config.update({"max_episode_steps": args["max_episode_steps"]})

    print("ENV CONFIG", env_config)

    # Create an env instance to introspect the gym spaces and episode length
    # when setting up the multiagent policy.
    env = env_creator(env_config)
    obs_space = env.observation_space
    act_space = env.action_space
    _ = env.reset()

    # Collect params related to train batch size and resources.
    rollout_fragment_length = env.max_episode_steps
    num_workers = args["num_cpus"]

    # Set any stopping conditions.
    stop = {
        'training_iteration': args["stop_iters"],
        'timesteps_total': args["stop_timesteps"],
        'episode_reward_mean': args["stop_reward"]
    }

    # Configure the deep learning framework.
    framework_config = {
        "framework": tune.grid_search([args["framework"]]),  # adds framework to trial name
        "eager_tracing": True  # ~3-4x faster than False
    }

    # Configure policy evaluation.  Evaluation appears to be broken using
    # pytorch, so consider omitting this.
    evaluation_config = {}
    if framework_config["framework"] == "tf2":
        evaluation_config = {
            "evaluation_interval": 1,
            "evaluation_num_episodes": 1,
            "evaluation_config": {"explore": False}
        }

    # Configure hyperparameters of the RL algorithm.
    hyperparam_config = {
        "lr": 1e-3,
        "num_sgd_iter": 10,
        "entropy_coeff": 0.0,
        "train_batch_size": rollout_fragment_length * 34,   # reproduces ours
        "rollout_fragment_length": rollout_fragment_length,
        "batch_mode": "complete_episodes",
        "observation_filter": "MeanStdFilter",
    }

    # Run the trial.
    experiment = tune.run(
        args["run"],
        local_dir=args["local_dir"],
        checkpoint_freq=1,
        checkpoint_at_end=True,
        checkpoint_score_attr="episode_reward_mean",
        keep_checkpoints_num=100,
        stop=stop,
        config={
            "env": env_name,
            "env_config": env_config,
            "num_gpus": args["num_gpus"],
            "num_workers": num_workers,
            "multiagent": {
                "policies": {
                    agent_id: (None, obs_space[agent_id], act_space[agent_id], {}) 
                        for agent_id in obs_space
                },
                "policy_mapping_fn": (lambda agent_id: agent_id)
            },
            "log_level": args["log_level"].upper(),
            **framework_config,
            **hyperparam_config,
            **evaluation_config
        }
    )

    return experiment


if __name__ == "__main__":

    from args import parser

    args = parser.parse_args()
    
    _ = main(**vars(args))

