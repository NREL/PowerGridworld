from datetime import datetime
import os
from typing import Tuple, Union, List

import numpy as np
import pandas as pd

from gridworld.log import logger
from gridworld.distribution_system.powerflow import PowerFlowSolver


DSS_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data/')


class OpenDSSSolver(PowerFlowSolver):

    def __init__(
        self,
        feeder_file: str,
        loadshape_file: str,    
        system_load_rescale_factor: float = 1.0,
        **kwargs
    ):
        """ An agent responsible for power flow computation given different 
        gen-load scenarios.

        Args:
            feeder_file: path to the dss file, relative to DSS_DATA_DIR.
            loadshape_file: path to the annual loadshape csv file, relative to 
                DSS_DATA_DIR.
            system_load_rescale_factor: scaling factor for base load
        """

        super().__init__(**kwargs)

        import opendssdirect as dss
        self.dss = dss
        self.dss_data_path = os.path.join(DSS_DATA_DIR, feeder_file)
        self.dss.run_command("Redirect " + self.dss_data_path)

        self.system_load_rescale_factor = system_load_rescale_factor

        load_profile_path = os.path.join(DSS_DATA_DIR, loadshape_file)
        self.annual_hourly_load_profile = np.genfromtxt(load_profile_path)
        if len(self.annual_hourly_load_profile) != 8760:
            print("Warning: The provided load shape file is not annual hourly ",
                "profile. Error might occur later")
             
        # Initialize class variable
        self.bus_voltages = {}
        self.load_bus_name, self.base_load = self._obtain_base_load_info()


    def _obtain_base_load_info(self) -> Tuple[list, np.ndarray]:
        """ Get base load info from the original OpenDSS project.

        Note, currently we only manipulate the PQ loads 
            (self.dss.Loads.Model() == 1).

        Returns:
          load_bus_name: A list of load bus names.
          base_load: Numpy array of (N, 2) dimension, where N is the system PQ 
            load number.
        """

        base_load = []
        load_bus_name = []

        ret = self.dss.Loads.First()
        while ret != 0:
            if self.dss.Loads.Model() == 1:
                base_load.append([self.dss.Loads.kW(), self.dss.Loads.kvar()])
                load_bus_name.append(self.dss.Loads.Name())
            ret = self.dss.Loads.Next()
        base_load = np.array(base_load)

        return load_bus_name, base_load


    def calculate_power_flow(
        self,
        p_controllable_consumed: dict = None,
        q_controllable_consumed: dict = None,
        current_time: str = None
    ) -> None:
        """ Calculate the power flow for the current time step.

        Args:
          p_controllable_consumed: dict of <bus, p>
          q_controllable_consumed: dict of <bus, q>
          current_time: string timestamp (convertible by pd.Timestamp)
        """

        # 1. Update the base load according the load shape file.

        current_time = pd.Timestamp(current_time)

        def get_hour_of_year(dt):
            """ Get hour of the year from pandas datetime object. Result is used 
            to retrieve load factor from the annual hourly load profile.
            """
            beginning_of_year = datetime(dt.year, 1, 1)
            return int((dt - beginning_of_year).total_seconds() // 3600)

        hour_of_year = get_hour_of_year(current_time)
        normalized_load_coefficient = self.annual_hourly_load_profile[hour_of_year]
        current_step_load = normalized_load_coefficient * self.base_load * \
            self.system_load_rescale_factor

        # 2. Update the PQ from uncontrollable assets at this step.
        # TODO: get this using current_time and uncontrollable assets profile

        # 3. Update the PQ from controllable assets at this step and direct set 
        # the number in OpenDSS.
        if p_controllable_consumed is not None:
            for idx, load_bus_name in enumerate(self.load_bus_name):

                try:
                    controllabe_p = p_controllable_consumed[load_bus_name]
                except KeyError:
                    controllabe_p = 0.0

                try:
                    controllabe_q = q_controllable_consumed[load_bus_name]
                except KeyError:
                    controllabe_q = 0.0

                current_step_load[idx, 0] += controllabe_p
                current_step_load[idx, 1] += controllabe_q

        self._set_load(current_step_load)

        # 4. Calculate the power flow
        self.dss.run_command('Solve mode=snap')
        self._prepare_bus_voltages()


    def _set_load(self, current_step_load) -> None:
        """ Set current load to OpenDSS.

        Args:
          current_step_load: Numpy array of (N, 2) dimension, where N is the 
            system PQ load number.
        """

        ret = self.dss.Loads.First()
        load_idx = 0
        while ret != 0:
            if self.dss.Loads.Model() == 1:  # Note, currently we only manipulate the PQ loads.
                self.dss.Loads.kW(current_step_load[load_idx, 0])
                self.dss.Loads.kvar(current_step_load[load_idx, 1])
                load_idx += 1
            ret = self.dss.Loads.Next()


    def _prepare_bus_voltages(self) -> None:
        """ Parse OpenDSS voltages to bus_name -> voltage mapping.

        Returns: None
        """
        voltages = self.dss.Circuit.AllBusMagPu()
        voltage_bus_name = self.dss.Circuit.AllNodeNames()

        for idx, bus_name in enumerate(voltage_bus_name):
            self.bus_voltages[bus_name] = voltages[idx]


    def get_bus_voltages(self) -> dict:
        """Returns a dict of <bus, voltages> on the feeder."""
        return self.bus_voltages


    def get_bus_voltage_by_name(self, bus_name: any) -> Union[float, List[float]]:
        """Returns the voltages at the specified bus.  If the bus is 3-phase,
        returns a list of floats, otherwise a single float."""

        PHASE_MAP = {'a': '.1', 'b': '.2', 'c': '.3'}

        if bus_name[-1] in PHASE_MAP.keys():
            bus_name = bus_name.replace(bus_name[-1], PHASE_MAP[bus_name[-1]])
            return self.bus_voltages[bus_name]
        else:
            # three phase bus, return voltage of all three phases.
            return [
                self.bus_voltages[x] 
                    for x in [bus_name + p for p in PHASE_MAP.values()]]


def main():

    dss_config = {"feeder_file": "ieee_13_dss/IEEE13Nodeckt.dss",
                  "loadshape_file": "ieee_13_dss/annual_hourly_load_profile.csv"}

    odss = OpenDSSSolver(**dss_config)
    current_time = pd.Timestamp("01-01-2021 05:00:00")
    odss.calculate_power_flow(current_time=current_time)
    v = odss.get_bus_voltages()

    import pprint
    pp = pprint.PrettyPrinter(indent=2)
    pp.pprint(v)


if __name__ == '__main__':
    main()

