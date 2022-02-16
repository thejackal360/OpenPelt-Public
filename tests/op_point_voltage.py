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
        os.mkdirs('./results/')
    pC = OpenPelt.plant_circuit("Detector",
                              lambda : 0.00@u_V,
                              OpenPelt.Signal.CURRENT)
    pC.characterize_plant(-6.00@u_V, 16.4@u_V, 0.01@u_V)
    pC.plot_th_tc(OpenPelt.IndVar.VOLTAGE)
    plt.savefig('./figs/{}'.format(TEST_NAME))
    data = numpy.array([pC.get_v_arr(), pC.get_th_actual()])
    numpy.save("./results/{}_volt_th_characterization".format(TEST_NAME), data)
    data = numpy.array([pC.get_v_arr(), pC.get_tc_actual()])
    numpy.save("./results/{}_volt_tc_characterization".format(TEST_NAME), data)
    plt.show()
