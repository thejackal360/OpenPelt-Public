#!/usr/bin/env python3

import matplotlib.pyplot as plt
import os
import OpenPelt
from PySpice.Unit import u_V
import time
import numpy

TEST_NAME = "volt_ref"

if __name__ == "__main__":
    if not os.path.exists('./results/'):
        os.makedirs('./results/')
    plate_select = OpenPelt.TECPlate.HOT_SIDE
    pC = OpenPelt.tec_plant("Detector",
                              lambda t, Th_arr: 0.0@u_V,
                              OpenPelt.Signal.VOLTAGE,
                              plate_select=plate_select)
    pC.run_sim()
    pC.plot_th_tc(OpenPelt.IndVar.TIME, plot_driver = False)
    plt.savefig('./figs/{}'.format(TEST_NAME))
    data = numpy.array([pC.get_t(), pC.get_th_sensor()])
    numpy.save("./results/{}_time_th_volt".format(TEST_NAME), data)
    data = numpy.array([pC.get_t(), pC.get_tc_sensor()])
    numpy.save("./results/{}_time_tc_volt".format(TEST_NAME), data)
    data = numpy.array([pC.get_t(), pC.get_v_arr()])
    numpy.save("./results/{}_time_volt".format(TEST_NAME), data)
    plt.savefig('./results/{}'.format(TEST_NAME))
