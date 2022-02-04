#!/usr/bin/env python3

import matplotlib.pyplot as plt
import os
import bessel
import time
import numpy

TEST_NAME = "mlp_hot"

if __name__ == "__main__":
    if not os.path.exists('./results/'):
        os.mkdirs('./results/')
    bessel.seed_everything(7777)
    plate_select = bessel.TECPlate.HOT_SIDE
    pC = bessel.plant_circuit("Detector", None, bessel.Signal.VOLTAGE, plate_select=plate_select)
    cbs = bessel.circular_buffer_sequencer([50.00, 30.00, 40.00], pC.get_ncs())
    nc = bessel.neural_controller(cbs, hidden_units = 6, lrate = 0.01, plate_select=plate_select)
    pC.set_controller_f(nc.controller_f)
    pC.run_sim()
    pC.plot_th_tc(bessel.IndVar.TIME, plot_driver = False, include_ref = True)
    plt.savefig('./figs/{}'.format(TEST_NAME))
    data = numpy.array([pC.get_t(), pC.get_th_sensor()])
    numpy.save('./results/{}_time_th_sensor_curr'.format(TEST_NAME), data)
    data = numpy.array([pC.get_t(), pC.get_tc_sensor()])
    numpy.save('./results/{}_time_tc_sensor_curr'.format(TEST_NAME), data)
    data = numpy.array([pC.get_t(), pC.get_v_arr()])
    numpy.save('./results/{}_time_v_curr'.format(TEST_NAME), data)
    plt.show()
