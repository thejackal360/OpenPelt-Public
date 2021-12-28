#!/usr/bin/env python3

from jinja2 import Template
import matplotlib.pyplot as plt
import numpy as np
import os

### Constants ###
SAMPLE_RATE = 44100 # Hz

### TEC model params class ###
class tec_pms:
    def __init__(self, l_tec, \
                       k_tec, \
                       a_tec, \
                       m_tec, \
                       c_tec, \
                       alpha, \
                       l_metal, \
                       k_metal, \
                       a_metal, \
                       m_metal, \
                       c_metal, \
                       rs, \
                       ambient_t, \
                       tc_tec):
        self.l_tec     = l_tec
        self.k_tec     = k_tec
        self.a_tec     = a_tec
        self.m_tec     = m_tec
        self.c_tec     = c_tec
        self.alpha     = alpha
        self.l_metal   = l_metal
        self.k_metal   = k_metal
        self.a_metal   = a_metal
        self.m_metal   = m_metal
        self.c_metal   = c_metal
        self.rs        = rs
        self.ambient_t = ambient_t
        self.tc_tec    = tc_tec

### Generate wav file ###
# data -> a numpy array of data
def gen_wav(t, data):
    assert len(t) == len(data)
    with open("output/inputvalues", "w") as inp:
        for i in range(len(t)):
            inp.write(str(t[i]) + " " + str(data[i]) + "\n")

### Generate params file ###
def gen_pms(pms_obj):
    with open("params.mod.template", "r") as p:
        t = Template(p.read())
        with open("output/params.mod", "w") as m:
            m.write(t.render(l_tec     = pms_obj.l_tec, \
                             k_tec     = pms_obj.k_tec, \
                             a_tec     = pms_obj.a_tec, \
                             m_tec     = pms_obj.m_tec, \
                             c_tec     = pms_obj.c_tec, \
                             alpha     = pms_obj.alpha, \
                             l_metal   = pms_obj.l_metal, \
                             k_metal   = pms_obj.k_metal, \
                             a_metal   = pms_obj.a_metal, \
                             m_metal   = pms_obj.m_metal, \
                             c_metal   = pms_obj.c_metal, \
                             rs        = pms_obj.rs, \
                             ambient_t = pms_obj.ambient_t, \
                             tc_tec    = pms_obj.tc_tec))

### Call ngspice ###
def call_ngspice():
    os.system("ngspice -b circuit.cir")

### Plot values ###
# TODO: Need real voltages
def get_values():
    with open("output/outputvalues", "r") as f:
        lines = [l.split() for l in f.read().split("\n")]
        t = [float(entry[0]) for entry in lines if entry != []]
        y = [float(entry[1]) for entry in lines if entry != []]
        return [t, y]

### Clean directory ###
# TODO

if __name__ == "__main__":
    import math
    # TEC1-12706
    tcp = tec_pms(l_tec = 3.60e-3, # [m], TEC thickness \
                  k_tec = 1.20, # [W/m*K], Bi2Te3 conductivity \
                  a_tec = 40.0e-3 * 40.0e-3, # [m^2], TEC surface area \
                  m_tec = 25e-3, # [kg], TEC mass (assume ceramic is most TEC mass) \
                  c_tec = 0.323e3, # [J/kg*K], ceramic specific heat cap \
                  alpha = 53e-3, # [V/K], Seebeck coefficient \
                  l_metal = 6.35e-3, # [m], 1/4in thick plate \
                  k_metal = 150.00, # [W/m*K], aluminum plate \
                  a_metal = 40.0e-3 * 40.0e-3, # [m^2], metal plate surface area \
                  m_metal = 2710.0 * 40.0e-3 * 40.0e-3 * 6.35e-3, # [kg], mass of plate \
                  c_metal = 0.89e3, # [J/kg*K], aluminum specific heat cap \
                  rs = 4.00, # [Ohm], TEC electrical resistance \
                  ambient_t = 27.00, # [C], ambient temp of ngspice sims \
                  tc_tec = 0.004) # [Ohm/K], temp coefficient of electrical resistivity
    t = np.linspace(10.00e-3, 1.00, 1000)
    y = len(t) * [4.00]
    # plt.plot(t, y)
    gen_wav(t, y)
    gen_pms(tcp)
    call_ngspice()
    [t0, y0] = get_values()
    plt.plot(t0, y0)
    plt.show()
