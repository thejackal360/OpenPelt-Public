#!/usr/bin/env python3
import numpy as np
import matplotlib.pylab as plt
from PySpice.Unit import u_V, u_A


from tecsim import test_control_algo
from initialize_randomness import seed_everything


def pretty_plot(data, ax=None, vmin=0.0, vmax=10.0):
    if ax is None:
        fig = plt.figure(figsize=(11, 5.5))
        fig.subplots_adjust(left=0.2, bottom=0.2)
        ax = fig.add_subplot(111)

    ax.plot(data[:, 0],
            data[:, 1],
            '-xk',
            lw=1.5,
            label="Hot temperature [C]")
    ax.plot(data[:, 0],
            data[:, 2],
            '-.xb',
            lw=1.5,
            label="Cold temperature [C]")
    ax.legend(fontsize=16)
    ax.set_xlabel("Voltage [V]", fontsize=18, weight='bold')
    ax.set_ylabel("Temperature [C]", fontsize=18, weight='bold')
    ax.set_xlim([vmin, vmax])
    ticks = ax.get_xticks()
    ax.set_xticklabels(ticks, fontsize=16, weight='bold')
    ticks = ax.get_yticks()
    ax.set_yticklabels(ticks, fontsize=16, weight='bold')
    ax.grid()


if __name__ == "__main__":
    seed_everything(7777)
    volt_ref, amp_ref = 0.0, 10.0

    def volt_input(t, Th_arr, Tc_arr):
        return volt_ref @ u_V

    def amp_input(t, Th_arr, Tc_arr):
        return amp_ref @ u_A

    res = []
    dt = .1
    step = int(10.0 / dt)
    for i in range(step):
        v, th, tc = test_control_algo(volt_input, voltage_src=True)
        volt_ref += dt
        res.append([v, th, tc])
        print("V: %f, Th: %f, Tc: %f" % (v, th, tc))
    res = np.array(res)
    np.save("./results/system_identification", res)

    pretty_plot(res, ax=None)
    plt.autoscale(enable=True, axis='x', tight=True)
    plt.savefig("figs/system_characterization.pdf")
    plt.show()
