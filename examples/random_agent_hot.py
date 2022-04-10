#!/usr/bin/env python3

import matplotlib.pyplot as plt
import os
import OpenPelt
import numpy

from OpenPelt.controller import random_agent_controller

from omnipyseed.seeding import universal_seed


TEST_NAME = "random_hot"


def random_agent(ref_temperatures=[85.00]):
    """
    Runs a simulation of a random agent as controller using OpenPelt library.
    This is a concept of proof demonstrating how one can use OpenPelt along
    with Reinforcement Learning algorithms to control TEC plates.

    Args:
        ref_temperatures (list):   A list of float numbers representing the
        reference temperatures provided by the user.

    Returns:
       A tuple that contains
       current (ndarray):  The current that drives the circuit over time (2, N)
       heat (ndarray):     The hot plate side temperature over time (2, N)
       coo(ndarray):       The cool plate side temperature over time (2, N)

       pC (OpenPelt object):   Provides all the methods implemented in OpenPelt
    """
    pC = OpenPelt.tec_plant("Detector", None, OpenPelt.Signal.VOLTAGE)
    cbs = OpenPelt.circular_buffer_sequencer([30.0], pC.get_ncs())
    nc = random_agent_controller(cbs)
    pC.set_controller_f(nc.controller_f)
    pC.run_sim()

    heat = numpy.array([pC.get_t(), pC.get_th_sensor()])
    cool = numpy.array([pC.get_t(), pC.get_tc_sensor()])
    voltage = numpy.array([pC.get_t(), pC.get_v_arr()])
    return (voltage, heat, cool), pC


if __name__ == "__main__":
    # check if the results directory exists otherwise create it
    if not os.path.exists('./results/'):
        os.mkdirs('./results/')

    # Seed all RNGs
    universal_seed(7777)

    # Run a simulation
    (voltage, heat, cool), pC = random_agent(ref_temperatures=[85.00])

    # Plot the results
    pC.plot_th_tc(OpenPelt.IndVar.TIME, plot_driver=False, include_ref=True)
    plt.savefig('./figs/{}'.format(TEST_NAME))

    # Store the data
    numpy.save('./results/{}_time_th_sensor_curr'.format(TEST_NAME), heat)
    numpy.save('./results/{}_time_tc_sensor_curr'.format(TEST_NAME), cool)
    numpy.save('./results/{}_time_v_curr'.format(TEST_NAME), voltage)
    plt.show()
