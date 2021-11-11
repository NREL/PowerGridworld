from abc import ABC, abstractmethod
from typing import Dict, Type

from gridworld.log import logger


class PowerFlowSolver(ABC):
    """Base class implementing the API for the power flow solver called from the
    MultiagentEnv environment."""

    def __init__(
        self,
        config: dict = None,
        **kwargs
    ):
        return


    @abstractmethod
    def calculate_power_flow(
        self,
        p_controllable_consumed: Dict[str, any] = None,
        q_controllable_consumed: Dict[str, any] = None,
        **kwargs
    ) -> any:
        """
        Compute the power flow solution using p/q consumed at each bus.

        Args:
            p_controllable_consumed: dict of (bus, real power) consumed.
            q_controllable_consumed: dict of (bus, reactive power) consumed.

        Returns: Any user-specified values.
        
        Whatever is done here needs to persist enough that the get_bus_voltages
        and get_bus_voltage_by_name methods return valid dicts.
        """

        raise NotImplementedError


    @abstractmethod
    def get_bus_voltages(self) -> Dict[str, any]:
        """Return a dict of (bus, voltage)."""
        raise NotImplementedError


    @abstractmethod
    def get_bus_voltage_by_name(self, name: str) -> any:
        """Return the voltage for a specific bus."""
        raise NotImplementedError