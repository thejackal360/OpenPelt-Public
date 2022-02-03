#!/usr/bin/env python3

import matplotlib.pyplot as plt
import os
import bessel
from PySpice.Unit import u_V
import time
import numpy

TEST_NAME = "transient"

if __name__ == "__main__":
    if not os.path.exists('./results/'):
        os.mkdirs('./results/')
    pC = bessel.plant_circuit("Detector",
                              lambda t, Th_arr: 0.0@u_V,
                              bessel.Signal.VOLTAGE)
    pC.run_sim()
    pC.plot_th_tc(bessel.IndVar.TIME)
    plt.savefig('./figs/{}'.format(TEST_NAME))
    data = numpy.array([pC.get_t(), pC.get_th_sensor()])
    numpy.save("./results/{}_time_th_volt".format(TEST_NAME), data)
    data = numpy.array([pC.get_t(), pC.get_tc_sensor()])
    numpy.save("./results/{}_time_tc_volt".format(TEST_NAME), data)
    data = numpy.array([pC.get_t(), pC.get_v_arr()])
    numpy.save("./results/{}_time_volt".format(TEST_NAME), data)
    plt.show()
