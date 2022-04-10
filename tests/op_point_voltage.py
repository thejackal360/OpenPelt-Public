#!/usr/bin/env python3

import matplotlib.pyplot as plt
import os
import OpenPelt
from PySpice.Unit import u_V
import numpy

TEST_NAME = "op_point_voltage"


def voltage_op_point():
    """
    Runs a simulation of an operational point using OpenPelt library.

    Args:

    Returns:
       A tuple that contains
       heat (ndarray):     The hot plate side temperature over voltage (2, N)
       coo(ndarray):       The cool plate side temperature over voltage (2, N)

       pC (OpenPelt object):   Provides all the methods implemented in OpenPelt
    """
    pC = OpenPelt.tec_plant("Detector",
                            lambda: 0.00@u_V,
                            OpenPelt.Signal.VOLTAGE)
    pC.characterize_plant(-5.00, 15.0, 0.1)

    heat = numpy.array([pC.get_v_arr(), pC.get_th_actual()])
    cool = numpy.array([pC.get_v_arr(), pC.get_tc_actual()])
    return (heat, cool), pC


if __name__ == "__main__":
    # Check if results directory exists otherwise create it
    if not os.path.exists('./results/'):
        os.makedirs('./results/')

    # Check if figs directory exists otherwise create it
    if not os.path.exists('./figs/'):
        os.makedirs('./figs/')

    # Run a simulation
    (heat, cool), pC = voltage_op_point()

    # Plot the characteristic curve of the TEC (temperature vs voltage)
    pC.plot_th_tc(OpenPelt.IndVar.VOLTAGE)
    plt.savefig('./figs/{}'.format(TEST_NAME))

    # Store the data
    numpy.save("./results/{}_volt_th_characterization".format(TEST_NAME), heat)
    numpy.save("./results/{}_volt_tc_characterization".format(TEST_NAME), cool)

    plt.show()
