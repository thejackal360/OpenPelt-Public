#!/usr/bin/env python3

import matplotlib.pyplot as plt
import os
import OpenPelt
import math

from OpenPelt.controller import pid_controller

TEST_NAME = "pid_hot"

CURR_T = 0.00
DT = 0.1

def next_sin():
    global CURR_T, DT
    CURR_T += DT
    output_val = math.sin(2.00 * math.pi * (1.00/100.00) * CURR_T)
    return output_val

if __name__ == "__main__":
    if not os.path.exists('./results/'):
        os.makedirs('./results/')
    if not os.path.exists('./figs/'):
        os.makedirs('./figs/')
    controller_f = lambda : next_sin()
    pC = OpenPelt.op_amp_plant("OpAmp", controller_f)
    pC.run_sim()
    pC.plot_output_v()
    plt.show()
