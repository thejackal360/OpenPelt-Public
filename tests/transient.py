#!/usr/bin/env python3

import matplotlib.pyplot as plt
import os
import OpenPelt
from PySpice.Unit import u_A
import numpy

TEST_NAME = "transient"


def transient():
    """
    Runs a transient simulation of the TEC model of the OpenPelt library.

    Args:

    Returns:
       A tuple that contains
       current (ndarray):  The current that drives the circuit over time (2, N)
       heat (ndarray):     The hot plate side temperature over time (2, N)
       coo(ndarray):       The cool plate side temperature over time (2, N)

       pC (OpenPelt object):   Provides all the methods implemented in OpenPelt
    """
    plate_select = OpenPelt.TECPlate.HOT_SIDE
    pC = OpenPelt.tec_plant("Detector",
                            lambda t, Th_arr: 2.1,
                            OpenPelt.Signal.CURRENT,
                            plate_select)
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

    # Run the transient simulation
    (current, heat, cool), pC = transient()

    # Plot the transient curve
    pC.plot_th_tc(OpenPelt.IndVar.TIME)
    plt.savefig('./figs/{}'.format(TEST_NAME))

    # Store the results
    numpy.save('./results/{}_time_th_sensor_curr'.format(TEST_NAME), heat)
    numpy.save('./results/{}_time_tc_sensor_curr'.format(TEST_NAME), cool)
    numpy.save('./results/{}_time_i_curr'.format(TEST_NAME), current)

    plt.savefig('./results/{}'.format(TEST_NAME))
    plt.show()
