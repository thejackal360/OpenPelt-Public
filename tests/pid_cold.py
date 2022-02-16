#!/usr/bin/env python3

import matplotlib.pyplot as plt
import os
import OpenPelt
import time
import numpy

TEST_NAME = "pid_cold"

if __name__ == "__main__":
    if not os.path.exists('./results/'):
        os.mkdirs('./results/')
    plate_select = OpenPelt.TECPlate.COLD_SIDE
    pC = OpenPelt.tec_plant("Detector", None, OpenPelt.Signal.VOLTAGE, plate_select=plate_select)
    cbs = OpenPelt.circular_buffer_sequencer([50.00, 30.00, 40.00], pC.get_ncs())
    pidc = OpenPelt.pid_controller(cbs, -30.00, -0.0007, -0.10, plate_select=plate_select)
    pC.set_controller_f(pidc.controller_f)
    pC.run_sim()
    pC.plot_th_tc(OpenPelt.IndVar.TIME, plot_driver = False, include_ref = True)
    plt.savefig('./figs/{}'.format(TEST_NAME))
    data = numpy.array([pC.get_t(), pC.get_th_sensor()])
    numpy.save('./results/{}_time_th_sensor_curr'.format(TEST_NAME), data)
    data = numpy.array([pC.get_t(), pC.get_tc_sensor()])
    numpy.save('./results/{}_time_tc_sensor_curr'.format(TEST_NAME), data)
    data = numpy.array([pC.get_t(), pC.get_v_arr()])
    numpy.save('./results/{}_time_v_curr'.format(TEST_NAME), data)
    plt.show()
