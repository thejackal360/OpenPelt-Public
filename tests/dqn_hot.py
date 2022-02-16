#!/usr/bin/env python3

import matplotlib.pyplot as plt
import os
import OpenPelt
import numpy
from torch import save, tensor


TEST_NAME = "dqn_hot"

if __name__ == "__main__":
    if not os.path.exists('./results/'):
        os.mkdirs('./results/')
    OpenPelt.seed_everything(7777)

    plate_select = OpenPelt.TECPlate.HOT_SIDE
    pC = OpenPelt.plant_circuit("Detector",
                              None,
                              OpenPelt.Signal.VOLTAGE,
                              plate_select=plate_select)
    cbs = OpenPelt.circular_buffer_sequencer([30.00],
                                           pC.get_ncs())
    nc = OpenPelt.dqn_controller(cbs,
                               gamma=0.999)
    pC.set_controller_f(nc.controller_f)

    pC.run_sim()

    pC.plot_th_tc(OpenPelt.IndVar.TIME, plot_driver=False, include_ref=True)
    plt.savefig('./figs/{}'.format(TEST_NAME))
    data = numpy.array([pC.get_t(), pC.get_th_sensor()])
    numpy.save('./results/{}_time_th_sensor_curr'.format(TEST_NAME), data)
    data = numpy.array([pC.get_t(), pC.get_tc_sensor()])
    numpy.save('./results/{}_time_tc_sensor_curr'.format(TEST_NAME), data)
    data = numpy.array([pC.get_t(), pC.get_v_arr()])
    numpy.save('./results/{}_time_v_curr'.format(TEST_NAME), data)
    plt.show()
