#!/usr/bin/env python3

import matplotlib.pyplot as plt
import os
import bessel
import numpy
from torch import save, tensor


TEST_NAME = "dqn_hot"

if __name__ == "__main__":
    if not os.path.exists('./results/'):
        os.mkdirs('./results/')
    # bessel.seed_everything(7777)

    ref_val = 24.0
    plate_select = bessel.TECPlate.HOT_SIDE
    pC = bessel.plant_circuit("Detector",
                              None,
                              bessel.Signal.VOLTAGE,
                              plate_select=plate_select)
    cbs = bessel.circular_buffer_sequencer([ref_val],
                                           pC.get_ncs())
    nc = bessel.dqn_controller(cbs,
                               gamma=0.99)
    n = 20
    for i in range(n):
        print("Episode #%d" % i)
        pC.set_controller_f(nc.controller_f)
        pC.run_sim()
        pC.clear()
        nc.reset()

        if i % 10 == 0:
            nc.target_net.load_state_dict(nc.policy_net.state_dict())
        if i >= 0 and i <= 5:
            colorH = 'k'
            colorC = 'y'
        elif i == n-1:
            colorH = 'orange'
            colorC = 'm'
        else:
            colorH = 'r'
            colorC = 'b'
        plt.plot(nc.th_hist, c=colorH, zorder=10)
        plt.plot(nc.tc_hist, c=colorC)
        plt.axhline(ref_val, ls='--', zorder=0)

        nc.num_episode += 1
        print("############################################################")
    save(nc.policy_net, "./results/policy_net.pt")
    save(nc.target_net, "./results/target_net.pt")

    # pC.plot_th_tc(bessel.IndVar.TIME, plot_driver=False, include_ref=True)
    # plt.savefig('./figs/{}'.format(TEST_NAME))
    # data = numpy.array([pC.get_t(), pC.get_th_sensor()])
    # numpy.save('./results/{}_time_th_sensor_curr'.format(TEST_NAME), data)
    # data = numpy.array([pC.get_t(), pC.get_tc_sensor()])
    # numpy.save('./results/{}_time_tc_sensor_curr'.format(TEST_NAME), data)
    # data = numpy.array([pC.get_t(), pC.get_v_arr()])
    # numpy.save('./results/{}_time_v_curr'.format(TEST_NAME), data)
    plt.show()
