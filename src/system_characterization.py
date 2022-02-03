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


if __name__ == '__main__':
    batch_size = 1
    epochs = 10

    v, tc = np.load("results/volt_tc_characterization.npy").astype('f')
    v, th = np.load("results/volt_th_characterization.npy").astype('f')

    c, tcc = np.load("results/curr_tc_characterization.npy").astype('f')
    c, tch = np.load("results/curr_th_characterization.npy").astype('f')

    fig = plt.figure(figsize=(13, 4))
    fig.subplots_adjust(left=0.1, bottom=0.2, wspace=0.2, hspace=0.2)
    ax = fig.add_subplot(121)
    ax.text(-5, 290, 'A',
            ha='left',
            fontsize=18,
            weight='bold')
    pretty_plot(c, tch, tcc, ax=ax, xlim=[-5, 5])
    ax = fig.add_subplot(122)
    pretty_plot(v, th, tc, ax=ax, ylabel=False, xlabel="volt", legend=False)
    ax.text(-5, 72, 'B',
            ha='left',
            fontsize=18,
            weight='bold')
    plt.savefig("figs/system_characterization.pdf")
    plt.show()
