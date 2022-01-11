#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy

if __name__ == "__main__":
    figure, ax = plt.subplots(figsize=(20, 10))
    _t = [60.00, 120.00, 180.00, 300.00, 360.00, 600.00, 840.00, 900.00, 1000.00]
    _y = [20.00 + 273.00, \
          15.00 + 273.00, \
          12.50 + 273.00, \
          7.50 + 273.00,  \
          5.00 + 273.00,  \
          4.00 + 273.00,  \
          2.00 + 273.00,  \
          1.00 + 273.00,  \
          0.00 + 273.00]
    ax.plot(_t, _y, label = 'Paper Result')
    with open("OUTPUT", "r") as o:
        _s = [s.split() for s in o.read().split('\n') if s != '']
        __t = [float(i[0]) for i in _s]
        __y = [float(i[1]) for i in _s]
        ax.plot(__t, __y, label = 'Sim Result')
    ax.set_xlabel('t')
    ax.set_ylabel('K')
    ax.legend()
    plt.show()
