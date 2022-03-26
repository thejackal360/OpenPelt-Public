#!/usr/bin/env python3

import matplotlib.pyplot as plt
import os
import OpenPelt
from PySpice.Unit import u_V
import time
import numpy

TEST_NAME = "op_point_voltage"

if __name__ == "__main__":
    if not os.path.exists('./results/'):
        os.makedirs('./results/')
    if not os.path.exists('./figs/'):
        os.makedirs('./figs/')
    pC = OpenPelt.tec_plant("Detector",
                              lambda : 0.00,
                              OpenPelt.Signal.VOLTAGE)
    pC.characterize_plant(-5.00, 15.0, 1.0)
    pC.plot_th_tc(OpenPelt.IndVar.VOLTAGE)
    plt.savefig('./figs/{}'.format(TEST_NAME))
    data = numpy.array([pC.get_v_arr(), pC.get_th_actual()])
    numpy.save("./results/{}_volt_th_characterization".format(TEST_NAME), data)
    data = numpy.array([pC.get_v_arr(), pC.get_tc_actual()])
    numpy.save("./results/{}_volt_tc_characterization".format(TEST_NAME), data)
    plt.savefig('./results/{}'.format(TEST_NAME))
    plt.show()
