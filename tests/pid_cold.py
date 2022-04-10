#!/usr/bin/env python3

import matplotlib.pyplot as plt
import os
import OpenPelt
import numpy

from OpenPelt.controller import pid_controller


TEST_NAME = "pid_cold"
TEMP_SENSOR_SAMPLES_PER_SEC = 1
SIMULATION_TIMESTEPS_PER_SENSOR_SAMPLE = 2


def pid_cold(ref_temperatures=[10.00, 15.00, 20.00, 25.00]):
    """
    Runs a simulation of a PID (hot plate side) controller using OpenPelt
    library. More details about the controller can be found in the file
    controller.py in OpenPelt directory.

    Args:
        ref_temperatures (list):   A list of float numbers representing the
        reference temperatures provided by the user.

    Returns:
       A tuple that contains
       voltage (ndarray):  The voltage that drives the circuit over time (2, N)
       heat (ndarray):     The hot plate side temperature over time (2, N)
       cool (ndarray):     The cool plate side temperature over time (2, N)

       pC (OpenPelt object):   Provides all the methods implemented in OpenPelt
    """
    plate_select = OpenPelt.TECPlate.COLD_SIDE
    pC = OpenPelt.tec_plant("Detector",
                            None,
                            OpenPelt.Signal.VOLTAGE,
                            plate_select=plate_select,
                            steady_state_cycles=400)
    cbs = OpenPelt.circular_buffer_sequencer([10.00, 15.00,
                                              20.00, 25.00],
                                              pC.get_ncs())
    pidc = OpenPelt.pid_controller(cbs, -150.00, 0.00, 0.00, \
                                   plate_select=plate_select)
    pC.set_controller_f(pidc.controller_f)
    pC.run_sim()

    heat = numpy.array([pC.get_t(), pC.get_th_sensor()])
    cool = numpy.array([pC.get_t(), pC.get_tc_sensor()])
    voltage = numpy.array([pC.get_t(), pC.get_v_arr()])
    return (voltage, heat, cool), pC


if __name__ == "__main__":
    # Check if directory results exists otherwise create it
    if not os.path.exists('./results/'):
        os.makedirs('./results/')

    # Check if directory figs exists otherwise create it
    if not os.path.exists('./figs/'):
        os.makedirs('./figs/')

    # Run the simulation
    (voltage, heat, cool), pC = pid_cold(ref_temperatures=[10.00,
                                                           15.00,
                                                           20.00,
                                                           25.00])

    # Plot the results
    pC.plot_th_tc(OpenPelt.IndVar.TIME, plot_driver=False, include_ref=True)
    plt.savefig('./figs/{}'.format(TEST_NAME))

    # Store the data
    numpy.save('./results/{}_time_th_sensor_curr'.format(TEST_NAME), heat)
    numpy.save('./results/{}_time_tc_sensor_curr'.format(TEST_NAME), cool)
    numpy.save('./results/{}_time_v_curr'.format(TEST_NAME), voltage)

    plt.savefig('./results/{}'.format(TEST_NAME))
    plt.show()
