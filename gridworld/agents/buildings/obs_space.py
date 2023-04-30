from typing import List, Dict
from collections import OrderedDict
from typing import Union, Tuple

import numpy as np
import gymnasium as gym


## DEFAULT BOUNDS for the observation space variables. Useful as reference.
# min/max temp that can be reached in the state space
ZONE_TEMP_BOUNDS = (16., 40.)
# min/max temp violation before clipping
ZONE_UPPER_TEMP_VIOL_BOUNDS = (-10., 10.)
ZONE_LOWER_TEMP_VIOL_BOUNDS = (-10., 10.)
# lower end of comfort temp goes between (a, b) throughout the day
COMFORT_TEMP_LOWER_BOUNDS = (20., 23.)
# upper end of comfort temp goes between (a, b) throughout the day
COMFORT_TEMP_UPPER_BOUNDS = (23., 26.)
# min/max outdoor temp
OUTDOOR_TEMP_BOUNDS = (0., 56.)
# min/max pq that can be reached
PQ_BOUNDS = (0., 200.)
# min/max bounds for time
TIME_BOUNDS = (0., 1.)
# bus voltages
BUS_VOLTAGE = (0.90, 1.10)

## DEFAULT CONFIGURATION.  This could be extended to include more than just the
# bounds (e.g., lookback horizon for outdoor temp, normalization params, etc).
DEFAULT_OBS_CONFIG = OrderedDict({
    "zone_temp": ZONE_TEMP_BOUNDS,    # temp per zone (C)
    "zone_upper_viol": ZONE_UPPER_TEMP_VIOL_BOUNDS,   # temp bound violation per zone (C)
    "zone_lower_viol": ZONE_LOWER_TEMP_VIOL_BOUNDS,   # temp bound violation per zone (C)
    "comfort_lower": COMFORT_TEMP_LOWER_BOUNDS,  # lower bound on comfort (C)
    "comfort_upper": COMFORT_TEMP_UPPER_BOUNDS,  # upper bound on comfort (C)
    "outdoor_temp": OUTDOOR_TEMP_BOUNDS, # outdoor temp (C) 
    "p_setpoint": PQ_BOUNDS,        # power setpoint (kwH)
    "p_consumed": PQ_BOUNDS,   # power consumed (kwH)
    "time_of_day": TIME_BOUNDS,   # normalized time of day (unitless between 0. and 1.),
    "bus_voltage": BUS_VOLTAGE,    # bus voltage,
    "min_voltage": BUS_VOLTAGE,    # min voltage on the network
    "max_voltage": BUS_VOLTAGE     # max voltage on the network
})

# These keys have one variable per zone.  This must be updated if additional
# keys are added to the default config, so the space maker knows which variables
# to expand to per-zone values.
MULTIZONE_KEYS = ["zone_temp", "zone_upper_viol", "zone_lower_viol"]


def _get_obs_dim(
        num_zones: int,
        config: Union[List, Dict]) -> int:
    """Returns the number of box dimensions for the given configuration params."""
    dim = 0
    for key in config:
        if key in MULTIZONE_KEYS:
            # These are zone-level variables
            dim += num_zones
        else:
            # The rest are scalar variables
            dim += 1
    return dim


def make_obs_space(
        num_zones:int, 
        config: dict) -> Tuple[gym.spaces.Box, List]:
    """Main function.  Returns a box space and list of dimension 
    labels for a given number of zones and configuration dict."""

    # Check that valid config keys were given
    for key in config:
        assert key in DEFAULT_OBS_CONFIG, "invalid key {}".format(key)

    # Generate min/max arrays for the obs space
    dim = _get_obs_dim(num_zones, config)
    obs_min = np.zeros(dim, dtype=float)
    obs_max = np.zeros(dim, dtype=float)
    obs_idx = 0  # current index 
    obs_labels = []   # string labels for each variable

    # Iterate through the keys and add their obs bounds to the array.
    for key in [k for k in DEFAULT_OBS_CONFIG if k in config]: # always uses same order
        if key in MULTIZONE_KEYS:
            # Here we are adding one variable per zone
            obs_min[obs_idx:obs_idx+num_zones] = config[key][0]
            obs_max[obs_idx:obs_idx+num_zones] = config[key][1]
            obs_idx += num_zones
            obs_labels.extend([key + "_" + str(i) for i in range(num_zones)])
        else:
            # Here we are adding just one scalar variable
            obs_min[obs_idx] = config[key][0]
            obs_max[obs_idx] = config[key][1]
            obs_idx += 1
            obs_labels.append(key)
        
    # Create the gym space
    obs_space = gym.spaces.Box(obs_min, obs_max, dtype=np.float64)

    return obs_space, obs_labels    


if __name__ == "__main__":

    def print_result(config, box, labels):
        print("CONFIG: ", config)
        print("LABELS: ", labels)
        print("SPACE: ", box)
        print("HIGH: ", box.high)
        print("LOW: ", box.low)
        print()

    num_zones = 5   # this is the only case we care about for now

    # A list of example obs space configurations
    test_configs = [
        # a dict lets you specify specific variables with custom min/max:
        {
            "zone_upper_viol": (-15, 15),
            "zone_lower_viol": (-15, 15),
            "comfort_lower": (22, 24),
            "comfort_upper": (25, 28)
        }
    ]

    # Generate an obs space for each configuration and print the result
    for config in test_configs:
        box, labels = make_obs_space(num_zones, config)
        print_result(config, box, labels)
