#!/usr/bin/env python3

import matplotlib.pyplot as plt
import os
import OpenPelt
from PySpice.Unit import u_A
import numpy

TEST_NAME = "op_point_current"


def current_op_point():
    """
    Runs a simulation of an operational point using OpenPelt library.

    Args:

    Returns:
       A tuple that contains
       heat (ndarray):     The hot plate side temperature vs current (2, N)
       coo(ndarray):       The cool plate side temperature vs current (2, N)

       pC (OpenPelt object):   Provides all the methods implemented in OpenPelt
    """
    pC = OpenPelt.tec_plant("Detector",
                            lambda: 2.1,
                            OpenPelt.Signal.CURRENT)
    pC.characterize_plant(-5.00, 5.00, 0.5)

    heat = numpy.array([pC.get_i_arr(), pC.get_th_actual()])
    cool = numpy.array([pC.get_i_arr(), pC.get_tc_actual()])
    return (heat, cool), pC


if __name__ == "__main__":
    # Check if results directory exists otherwise create it
    if not os.path.exists('./results/'):
        os.makedirs('./results/')

    # Check if figs directory exists otherwise create it
    if not os.path.exists('./figs/'):
        os.makedirs('./figs/')

    # Run a simulation
    (heat, cool), pC = current_op_point()

    # Plot the characteristic curve of TEC temperature vs current
    pC.plot_th_tc(OpenPelt.IndVar.CURRENT)
    plt.savefig('./figs/{}'.format(TEST_NAME))

    # Store the data
    numpy.save("./results/{}_curr_th_characterization".format(TEST_NAME), heat)
    numpy.save("./results/{}_curr_tc_characterization".format(TEST_NAME), cool)
    plt.savefig('./results/{}'.format(TEST_NAME))
    plt.show()
