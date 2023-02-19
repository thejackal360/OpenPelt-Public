#!/usr/bin/env python3

import matplotlib.pyplot as plt
import os
import OpenPelt
import numpy

from OpenPelt.controller import bang_bang_controller

TEST_NAME = "basic_bang_bang"


def bang_bang(reference_temperatures=[50.00, 30.00]):
    """
    Runs a simulation of a Bang-bang controller using OpenPelt library. More
    details about the controller can be found in the file controller.py in
    OpenPelt directory.

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
    pC = OpenPelt.tec_plant("Detector", None, OpenPelt.Signal.VOLTAGE,
                            steady_state_cycles=400)
    cbs = OpenPelt.circular_buffer_sequencer(reference_temperatures,
                                             pC.get_ncs())
    bbc = bang_bang_controller(cbs)
    pC.set_controller_f(bbc.controller_f)
    pC.run_sim()

    heat = numpy.array([pC.get_t(), pC.get_th_sensor()])
    cool = numpy.array([pC.get_t(), pC.get_tc_sensor()])
    current = numpy.array([pC.get_t(), pC.get_i_arr()])

    return (current, heat, cool), pC


if __name__ == "__main__":
    # Check if directory results exists otherwise create it
    if not os.path.exists('./results/'):
        os.makedirs('./results/')

    # Check if directory figs exists otherwise create it
    if not os.path.exists('./figs/'):
        os.makedirs('./figs/')

    # Run the simulation of a Bang-bang controller
    (current, heat, cool), pC = bang_bang([50.00, 30.00])

    # Plot the results using the plot_th_th method of OpenPelt
    pC.plot_th_tc(OpenPelt.IndVar.TIME, plot_driver=False, include_ref=True)
    plt.savefig('./figs/{}'.format(TEST_NAME))

    # Store all the data for further analysis
    numpy.save('./results/{}_time_th_sensor_curr'.format(TEST_NAME), heat)
    numpy.save('./results/{}_time_tc_sensor_curr'.format(TEST_NAME), cool)
    numpy.save('./results/{}_time_i_curr'.format(TEST_NAME), current)

    plt.savefig('./results/{}'.format(TEST_NAME))
    plt.show()
