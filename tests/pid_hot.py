#!/usr/bin/env python3

import matplotlib.pyplot as plt
import os
import OpenPelt
import numpy

from OpenPelt.controller import pid_controller

TEST_NAME = "pid_hot"
TEMP_SENSOR_SAMPLES_PER_SEC = 1
SIMULATION_TIMESTEPS_PER_SENSOR_SAMPLE = 2


if __name__ == "__main__":
    if not os.path.exists('./results/'):
        os.makedirs('./results/')
    if not os.path.exists('./figs/'):
        os.makedirs('./figs/')
    plate_select = OpenPelt.TECPlate.HOT_SIDE
    pC = OpenPelt.tec_plant("Detector",
                            None,
                            OpenPelt.Signal.VOLTAGE,
                            plate_select=plate_select,
                            steady_state_cycles=400)
    cbs = OpenPelt.circular_buffer_sequencer([30.00, 40.00, 50.00],
                                             pC.get_ncs())
    pidc = pid_controller(cbs, 15.00, 0.00, 0.00,
                          plate_select,
                          TEMP_SENSOR_SAMPLES_PER_SEC,
                          SIMULATION_TIMESTEPS_PER_SENSOR_SAMPLE)
    pC.set_controller_f(pidc.controller_f)
    pC.run_sim()
    pC.plot_th_tc(OpenPelt.IndVar.TIME, plot_driver=False, include_ref=True)
    plt.savefig('./figs/{}'.format(TEST_NAME))
    data = numpy.array([pC.get_t(), pC.get_th_sensor()])
    numpy.save('./results/{}_time_th_sensor_curr'.format(TEST_NAME), data)
    data = numpy.array([pC.get_t(), pC.get_tc_sensor()])
    numpy.save('./results/{}_time_tc_sensor_curr'.format(TEST_NAME), data)
    data = numpy.array([pC.get_t(), pC.get_v_arr()])
    numpy.save('./results/{}_time_v_curr'.format(TEST_NAME), data)
    plt.savefig('./results/{}'.format(TEST_NAME))
    plt.show()
