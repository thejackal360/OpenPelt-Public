#!/usr/bin/env python3

import matplotlib.pyplot as plt
import os
import bessel
import time
import numpy

TEST_NAME = "basic_bang_bang"

if __name__ == "__main__":
    if not os.path.exists('./results/'):
        os.mkdirs('./results/')
    pC = bessel.plant_circuit("Detector", None, bessel.Signal.VOLTAGE)
    cbs = bessel.circular_buffer_sequencer([50.00, 30.00], pC.get_ncs())
    bbc = bessel.bang_bang_controller(cbs)
    pC.set_controller_f(bbc.controller_f)
    pC.run_sim()
    pC.plot_th_tc(bessel.IndVar.TIME, plot_driver = False, include_ref = True)
    plt.savefig('./figs/{}'.format(TEST_NAME))
    data = numpy.array([pC.get_t(), pC.get_th_sensor()])
    numpy.save('./results/{}_time_th_sensor_curr'.format(TEST_NAME), data)
    data = numpy.array([pC.get_t(), pC.get_tc_sensor()])
    numpy.save('./results/{}_time_tc_sensor_curr'.format(TEST_NAME), data)
    data = numpy.array([pC.get_t(), pC.get_i_arr()])
    numpy.save('./results/{}_time_i_curr'.format(TEST_NAME), data)
    plt.show()
