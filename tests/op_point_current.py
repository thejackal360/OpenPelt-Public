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
                            lambda: 2.1,
                            OpenPelt.Signal.CURRENT)
    pC.characterize_plant(-5.00, 5.00, 0.5)
    pC.plot_th_tc(OpenPelt.IndVar.CURRENT)
    plt.savefig('./figs/{}'.format(TEST_NAME))
    data = numpy.array([pC.get_i_arr(), pC.get_th_actual()])
    numpy.save("./results/{}_curr_th_characterization".format(TEST_NAME), data)
    data = numpy.array([pC.get_i_arr(), pC.get_tc_actual()])
    numpy.save("./results/{}_curr_tc_characterization".format(TEST_NAME), data)
    plt.savefig('./results/{}'.format(TEST_NAME))
    plt.show()
