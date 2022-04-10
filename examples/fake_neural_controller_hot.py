#!/usr/bin/env python3

import matplotlib.pyplot as plt
import os
import OpenPelt
import numpy

from OpenPelt.controller import fake_neural_controller

from omnipyseed.seeding import universal_seed


TEST_NAME = "random_hot"


def neural_network_agent(ref_temperature=30.00):
    """
    Runs a simulation of an untrained neural network as a controller using
    OpenPelt library.
    This is a concept of proof demonstrating how one can use OpenPelt along
    with Neural Networks and PyTorch to control TEC plates.

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
    nc = fake_neural_controller(cbs)
    pC.set_controller_f(nc.controller_f)
    pC.run_sim()

    heat = numpy.array([pC.get_t(), pC.get_th_sensor()])
    cool = numpy.array([pC.get_t(), pC.get_tc_sensor()])
    voltage = numpy.array([pC.get_t(), pC.get_v_arr()])

    return (voltage, heat, cool), pC


if __name__ == "__main__":
    if not os.path.exists('./results/'):
        os.mkdirs('./results/')

    # Seed everything
    universal_seed(7777)

    # Run a simulation
    (voltage, heat, cool), pC = neural_network_agent(ref_temperature=30.00)

    # Plot the results using the plot method of OpenPelt
    pC.plot_th_tc(OpenPelt.IndVar.TIME, plot_driver=False, include_ref=True)
    plt.savefig('./figs/{}'.format(TEST_NAME))

    # Store all the data for further analysis
    numpy.save('./results/{}_time_th_sensor_curr'.format(TEST_NAME), heat)
    numpy.save('./results/{}_time_tc_sensor_curr'.format(TEST_NAME), cool)
    numpy.save('./results/{}_time_v_curr'.format(TEST_NAME), voltage)

    plt.show()
