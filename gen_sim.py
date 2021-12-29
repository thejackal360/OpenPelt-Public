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
                       tc_tec, \
                       delta_t_0, \
                       i_0, \
                       h_tec, \
                       alpha_c):
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
        self.delta_t_0 = delta_t_0
        self.i_0       = i_0
        self.h_tec     = h_tec
        self.alpha_c   = alpha_c

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
                             tc_tec    = pms_obj.tc_tec, \
                             delta_t_0 = pms_obj.delta_t_0, \
                             i_0       = pms_obj.i_0, \
                             h_tec     = pms_obj.h_tec, \
                             alpha_c   = pms_obj.alpha_c))

### Call ngspice ###
def call_ngspice():
    os.system("ngspice circuit.cir")
    # os.system("ngspice -b circuit.cir")

### Plot values ###
# TODO: Need real voltages
def get_values(peltier, Rs):
    with open("output/tecplus_output") as tecplus_f:
        with open("output/seebeck_output") as seebeck_f:
            tecplus_lines = [l.split() for l in tecplus_f.read().split("\n")]
            tecplus_t = [float(entry[0]) for entry in tecplus_lines if entry != []]
            tecplus_y = [float(entry[1]) for entry in tecplus_lines if entry != []]
            seebeck_lines = [l.split() for l in seebeck_f.read().split("\n")]
            seebeck_t = [float(entry[0]) for entry in seebeck_lines if entry != []]
            seebeck_y = [float(entry[1]) for entry in seebeck_lines if entry != []]
            return peltier * (tecplus_y[len(tecplus_y)-1] - seebeck_y[len(seebeck_y)-1])/Rs
    # with open("output/th_output", "r") as th_f:
    #     with open("output/tc_output", "r") as tc_f:
    #         th_lines = [l.split() for l in th_f.read().split("\n")]
    #         th_t = [float(entry[0]) for entry in th_lines if entry != []]
    #         th_y = [float(entry[1]) for entry in th_lines if entry != []]
    #         tc_lines = [l.split() for l in tc_f.read().split("\n")]
    #         tc_t = [float(entry[0]) for entry in tc_lines if entry != []]
    #         tc_y = [float(entry[1]) for entry in tc_lines if entry != []]
    #         return th_y[len(th_y)-1] - tc_y[len(tc_y)-1]

def get_delta_t(V):
    import math
    # TEC1-12706
    tcp = tec_pms(l_tec = 3.60e-3 / 2.00, ### 3.60e-3, # [m], TEC thickness \
                  k_tec = 1.20, # [W/m*K], Bi2Te3 conductivity \
                  a_tec = 40.0e-3 * 40.0e-3, # [m^2], TEC surface area \
                  m_tec = 25e-3 / 2.00, ### 25e-3, # [kg], TEC mass (assume ceramic is most TEC mass) \
                  c_tec = 0.323e3, # [J/kg*K], ceramic specific heat cap \
                  alpha = 53e-3, # [V/K], Seebeck coefficient \
                  l_metal = 6.35e-3, # [m], 1/4in thick plate \
                  k_metal = 150.00, # [W/m*K], aluminum plate \
                  a_metal = 40.0e-3 * 40.0e-3, # [m^2], metal plate surface area \
                  m_metal = 2710.0 * 40.0e-3 * 40.0e-3 * 6.35e-3, # [kg], mass of plate \
                  c_metal = 0.89e3, # [J/kg*K], aluminum specific heat cap \
                  rs = 2.10, # [Ohm], TEC electrical resistance \
                  ambient_t = 27.00, # [C], ambient temp of ngspice sims \
                  tc_tec = 0.004, # [Ohm/K], temp coefficient of electrical resistivity \
                  delta_t_0 = 30.00, # [K], reference operating point temperature difference \
                  i_0 = 2.50, # [A], reference operating point current \
                  h_tec = 12.12, # [W/m2*K], convective heat transfer coefficient \
                  alpha_c = 0.15) # [V/K^2], Seebeck temperature coefficient
    t = np.linspace(10.00e-3, 1.00, 1000)
    y = len(t) * [V]
    # plt.plot(t, y)
    gen_wav(t, y)
    gen_pms(tcp)
    call_ngspice()
    final_qc = get_values((tcp.k_tec * tcp.delta_t_0) / tcp.i_0, tcp.rs)
    return final_qc

### Clean directory ###
# TODO

if __name__ == "__main__":
    _V = []
    _delta_T = []
    for V in range(30):
        _V.append(V)
        _delta_T.append(get_delta_t(float(V)))
    plt.xlabel("Voltage [V]")
    plt.ylabel("Qc [W]")
    plt.plot(_V, _delta_T)
    plt.show()
    # plt.plot(t0, y0)
    # plt.show()
