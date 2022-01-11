#!/usr/bin/env python3

from PySpice.Spice.Netlist import Circuit, SubCircuit
from PySpice.Spice.HighLevelElement import PulseCurrentSource
from PySpice.Unit import *

import matplotlib.pyplot as plt
import numpy

TAMB         = 296.4
TAMB_V       = True
RP           = 1.8
SE           = 0.05292
K_RAD        = 0.34
C_RAD        = 340.00
K_SIL        = 0.143
C_H          = 2.00
K_M          = 1.768
C_C          = 2.00
C_CONINT     = 304.00
K_CONINT     = 3.1
PULSED_VALUE = 2.1

cir = Circuit('MyCirc')
### HEAT SINK
if TAMB_V:
    cir.V('1', '3', cir.gnd, TAMB@u_V)
    cir.R('1', '4', '3', K_RAD@u_Ohm)
    cir.C('1', '4', cir.gnd, C_RAD@u_F, initial_condition = TAMB@u_V)
    cir.R('2', '4', '1', K_SIL@u_Ohm)
### THERMAL PELTIER MODEL
cir.C('2', '1', '0', C_H@u_F, initial_condition = TAMB@u_V)
cir.BehavioralSource('1', cir.gnd, '1', \
        i = '((v(13) - v(12))/{})*(((v(13) - v(12))/{})*{}+{}*(v(1)-v(2)))'.format(RP, RP, RP, SE))
cir.R('3', '1', '2', K_M@u_Ohm)
cir.BehavioralSource('2', '2', '1', \
        i = '((v(13) - v(12))/{})*({}*v(2)-0.9*((v(13) - v(12))/{}))'.format(RP, SE, RP))
cir.C('3', '2', cir.gnd, C_C@u_F, initial_condition = TAMB@u_V)
### THERMAL MASS
if TAMB_V:
    cir.R('4', '5', '2', K_SIL@u_Ohm)
    cir.C('4', '5', cir.gnd, C_CONINT@u_F, initial_condition = TAMB@u_V)
    cir.R('5', '5', '3', K_CONINT@u_Ohm)
### ELECTRICAL PELTIER MODEL
cir.V('2', '11', '13', 0.00@u_V)
cir.R('6', '13', '12', RP@u_Ohm)
cir.VCVS('1', '12', cir.gnd, '1', '2', voltage_gain = SE)
### EXTERNAL CURRENT SOURCE
PulseCurrentSource(cir, '1', cir.gnd, '11', pulsed_value = 2.1@u_A, pulse_width = 1e6@u_s, period = 1e7@u_s, initial_value = 0.00@u_A)
simulator = cir.simulator()
simulator.options(reltol = 5e-6)
analysis = simulator.transient(step_time = 1.00@u_s, end_time = 1800.00@u_s, use_initial_condition = True)
figure, ax = plt.subplots(figsize=(20, 10))
ax.plot(analysis['1'], label = 'Th Sim Result')
ax.plot(analysis['2'], label = 'Tc Sim Result')
_th = [1.00 * 60.00, \
       2.00 * 60.00, \
       3.00 * 60.00, \
       5.00 * 60.00, \
       6.00 * 60.00, \
       10.00 * 60.00, \
       14.00 * 60.00, \
       15.00 * 60.00, \
       16.00 * 60.00, \
       20.00 * 60.00]
_yh = [28.00 + 273.00, \
       31.00 + 273.00, \
       32.50 + 273.00, \
       34.00 + 273.00, \
       33.50 + 273.00, \
       32.50 + 273.00, \
       32.00 + 273.00, \
       31.50 + 273.00, \
       31.50 + 273.00, \
       31.50 + 273.00]
ax.plot(_th, _yh, label = 'Th Paper Result')
_tc = [60.00, 120.00, 180.00, 300.00, 360.00, 600.00, 840.00, 900.00, 1000.00]
_yc = [20.00 + 273.00, \
      15.00 + 273.00, \
      12.50 + 273.00, \
      7.50 + 273.00,  \
      5.00 + 273.00,  \
      4.00 + 273.00,  \
      2.00 + 273.00,  \
      1.00 + 273.00,  \
      0.00 + 273.00]
ax.plot(_tc, _yc, label = 'Tc Paper Result')
ax.set_xlabel('t')
ax.set_ylabel('K')
ax.legend()
plt.show()
