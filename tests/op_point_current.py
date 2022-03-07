#!/usr/bin/env python3

import matplotlib.pyplot as plt
import os
import OpenPelt
from PySpice.Unit import u_A
import numpy

TEST_NAME = "op_point_current"

if __name__ == "__main__":
    if not os.path.exists('./results/'):
        os.makedirs('./results/')
    if not os.path.exists('./figs/'):
        os.makedirs('./figs/')
    pC = OpenPelt.tec_plant("Detector",
                            lambda: 2.1@u_A,
                            OpenPelt.Signal.CURRENT)
    pC.characterize_plant(-6.00@u_A, 6.00@u_A, 0.01@u_A)
    pC.plot_th_tc(OpenPelt.IndVar.CURRENT)
    plt.savefig('./figs/{}'.format(TEST_NAME))
    data = numpy.array([pC.get_i_arr(), pC.get_th_actual()])
    numpy.save("./results/{}_curr_th_characterization".format(TEST_NAME), data)
    data = numpy.array([pC.get_i_arr(), pC.get_tc_actual()])
    numpy.save("./results/{}_curr_tc_characterization".format(TEST_NAME), data)
    plt.savefig('./results/{}'.format(TEST_NAME))
    plt.show()
