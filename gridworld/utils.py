import logging

import numpy as np

import gymnasium as gym

from gridworld.log import logger


def to_scaled(
    x: np.ndarray,
    low: np.ndarray,
    high: np.ndarray
) -> np.ndarray:
    """Scale the input arr in [low, high] to [-1, 1]"""

    # Warn the user if the arguments are out of bounds, this shouldn't happend.
    if not np.all(x >= low) and np.all(x <= high):
        logger.warning(f"argument out of bounds, {x}, {low}, {high}")
    
    # Clip the values (in case the above warning is ignored).
    x = np.clip(x, low, high)
    
    # Transform the input to [-1, 1].
    return (2*x - (low + high)) / (high - low)


def to_raw(
    y: np.ndarray,
    low: np.ndarray,
    high: np.ndarray,
    eps: float = 1e-4
) -> np.ndarray:
    """Scale the input y in [-1, 1] to [low, high]"""

    # Warn the user if the arguments are out of bounds, this shouldn't happend.""""
    if not (np.all(y >= -np.ones_like(y) - eps) and np.all(y <= np.ones_like(y) + eps)):
        logger.warning(f"argument out of bounds, {y}, {low}, {high}")
    
    # Clip the values (in case the above warning is ignored).
    y = np.clip(y, -np.ones_like(y), np.ones_like(y))
    
    # Transform the input to [low, high].
    return (y * (high - low) + (high + low)) / 2.


def maybe_rescale_box_space(box, rescale=True):
    """Create a new box space with bounds [-1, 1] and same shape, type as input
    space.  Allow calling as a pass-through with rescale=False."""

    if rescale:
        return gym.spaces.Box(low=-1., high=1., shape=box.shape, dtype=box.dtype)
    
    return box


# def multiagent_action_rescaler(
#     rescale_func: callable,
#     env: MultiAgentEnv = None,
#     action: dict = None,
# ):
#     rescaled_action = {}
#     for a in env.agent_dict:
#         rescaled_action[a] = {}
#         for e in env.agent_dict[a].env_dict:
#             _env = env.agent_dict[a].env_dict[e]
#             rescaled_action[a][e] = rescale_func(
#                 action[a][e],
#                 _env._action_space.low,
#                 _env._action_space.high)
#     return rescaled_action




