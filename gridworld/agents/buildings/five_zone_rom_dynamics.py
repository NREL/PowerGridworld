"""Implements the time-variant linear dynamics for five-zone Reduced Order
Model (ROM)."""

from typing import Tuple

import numpy as np


Z = 5    # number of zones


def build_u_vector(
    models: dict,
    zone_temp: np.ndarray,
    action: np.ndarray,
    temp_oa: np.ndarray,
    q_solar: np.ndarray,
    q_int: np.ndarray,
    q_cool: np.ndarray = None
) -> np.ndarray:
    """Returns the u-vector based on temps, actions, and exogenous data.
    If at is None, q_cool must be provided."""
    
    u_pos = np.zeros((Z, 8), dtype=np.float64)
    for z in range(Z):
        u_pos[z, 0] = temp_oa - zone_temp[z]
        u_pos[z, 1] = q_solar[z]
        u_pos[z, 2] = q_int[z]

        for i, y in enumerate(models[z]["neighbors"]):
            u_pos[z, 3+i] = zone_temp[y] - zone_temp[z]

        u_pos[z, 7] = q_cool[z] if action is None else \
            action[z] * (action[-1] - zone_temp[z])

    u = np.zeros((Z, 4), dtype=np.float64)
    for z in range(Z):
        for u_idx, sel_idx in enumerate(models[z]["input_sel_list"][0]):
            u[z, u_idx] = u_pos[z][sel_idx - 1]

    return u


def state_update(models: dict, u: np.ndarray) -> dict:
    """Returns the next state based on current state and u-vector."""
    
    for z in range(Z):
        models[z]["x_k"] = \
            models[z]["ss_A"].squeeze() * models[z]["x_k"].squeeze() + \
            np.matmul(
                np.array(models[z]["ss_B"]).astype(np.float32).reshape(1, -1), 
                u[z, :].reshape(-1, 1)
            )

    return models


def filter_update(
    models: dict, 
    zone_temp: np.ndarray,
    u: np.ndarray
) -> np.ndarray:
    """Perform the filter update given current state, temps, and u-vector."""

    models = state_update(models, u)

    for z in range(Z):
        yhat_kplus1 = models[z]["ss_C"][0][0] * models[z]["x_k"].squeeze()
        y_actual = zone_temp[z] - models[z]["mean_output"]
        models[z]["x_k"] += models[z]["ss_K"].squeeze() * (y_actual - yhat_kplus1)

    return models


def temp_dynamics(
    models: dict,
) -> np.ndarray:
    """Returns the new zone temperatures based on the current state."""

    zone_temp = np.zeros(Z, dtype=np.float64)
    for z in range(Z):
        y = np.array(models[z]["ss_C"]).squeeze() * models[z]["x_k"].squeeze()
        zone_temp[z] = y + models[z]["mean_output"].squeeze()

    return zone_temp


def dynamics(
    models: dict,
    zone_temp: np.ndarray,
    action: np.ndarray,
    temp_oa: np.ndarray, 
    q_solar: np.ndarray, 
    q_int: np.ndarray
) -> Tuple[dict, np.ndarray, np.ndarray]:
    """returns the new state and zone temps based on current state, temps, actions
    and exogenous data."""

    u = build_u_vector(models, zone_temp, action, temp_oa, q_solar, q_int)
    models = state_update(models, u)
    zone_temp = temp_dynamics(models)

    return models, zone_temp


def get_p_consumed(
    action: np.ndarray,
    temp_oa: np.ndarray
) -> np.ndarray:

    fan_power = 0.0076 * np.sum(action[:-1])**3 + 4.8865
    chiller_power = max(0., np.sum(action[:-1]) * (temp_oa - action[-1]))

    return fan_power + chiller_power


def stage_cost(
    zone_temp: np.ndarray,
    action: np.ndarray,
    temp_oa: float,
    comfort_bounds: tuple,
    tou: float,
    comfort_penalty: float,
) -> float :
    """Returns the stage cost given the action, temp violations, and tou price."""

    p_consumed = get_p_consumed(action, temp_oa)
    power_cost = tou * p_consumed

    vl = max(np.maximum(np.zeros_like(zone_temp), comfort_bounds[0] - zone_temp))
    vu = max(np.maximum(np.zeros_like(zone_temp), zone_temp - comfort_bounds[1]))
    viol_cost = comfort_penalty * np.sum(vl + vu)

    return power_cost + viol_cost



