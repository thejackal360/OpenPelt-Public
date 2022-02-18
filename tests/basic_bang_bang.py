#!/usr/bin/env python3

import matplotlib.pyplot as plt
import os
import OpenPelt
import time
import numpy

TEST_NAME = "basic_bang_bang"

if __name__ == "__main__":
    if not os.path.exists('./results/'):
        os.makedirs('./results/')
    pC = OpenPelt.tec_plant("Detector", None, OpenPelt.Signal.VOLTAGE)
    cbs = OpenPelt.circular_buffer_sequencer([50.00, 30.00], pC.get_ncs())
    bbc = OpenPelt.bang_bang_controller(cbs)
    pC.set_controller_f(bbc.controller_f)
    pC.run_sim()
    pC.plot_th_tc(OpenPelt.IndVar.TIME, plot_driver = False, include_ref = True)
    plt.savefig('./figs/{}'.format(TEST_NAME))
    data = numpy.array([pC.get_t(), pC.get_th_sensor()])
    numpy.save('./results/{}_time_th_sensor_curr'.format(TEST_NAME), data)
    data = numpy.array([pC.get_t(), pC.get_tc_sensor()])
    numpy.save('./results/{}_time_tc_sensor_curr'.format(TEST_NAME), data)
    data = numpy.array([pC.get_t(), pC.get_i_arr()])
    numpy.save('./results/{}_time_i_curr'.format(TEST_NAME), data)
    plt.savefig('./results/{}'.format(TEST_NAME))
