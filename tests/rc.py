#!/usr/bin/env python3

import matplotlib.pyplot as plt
import os
import OpenPelt
from PySpice.Unit import u_V

TEST_NAME = "rc"

if __name__ == "__main__":
    if not os.path.exists('./results/'):
        os.makedirs('./results/')
    if not os.path.exists('./figs/'):
        os.makedirs('./figs/')
    pC = OpenPelt.rc_ckt_plant("RC", None)
    pC.set_controller_f(lambda : 1.00)
    pC.run_sim()
    pC.plot_output_v()
    plt.show()
