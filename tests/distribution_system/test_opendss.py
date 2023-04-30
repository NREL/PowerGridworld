from gridworld.distribution_system import OpenDSSSolver


def test_default_opendss(opendss_config):

    opendss = OpenDSSSolver(**opendss_config)

    current_time = "01-01-2021 05:00:00"
    opendss.calculate_power_flow(current_time=current_time)
    
    assert opendss.get_bus_voltages()
