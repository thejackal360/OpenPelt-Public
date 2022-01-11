#!/usr/bin/env python3

from PySpice.Spice.Netlist import Circuit, SubCircuit
from PySpice.Spice.NgSpice.Shared import NgSpiceShared
from PySpice.Unit import *
import matplotlib.pyplot as plt
import os

# TODO: return sensor reading in C
# TODO: check ngspice version
# TODO: why is hot side colder than cold side?

### Detector Circuit Parameters ###
TAMB         = 296.4
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

class NgspiceCustomSrc(NgSpiceShared):
    def __init__(self, f_of_t_and_T, **kwargs):
        super().__init__(**kwargs)
        self.f_of_t_and_T = f_of_t_and_T
        self.last_th = TAMB
    def send_data(self, actual_vector_values, number_of_vectors, ngspice_id):
        self.last_th = actual_vector_values['V(1)']
        return 0
    def get_vsrc_data(self, voltage, time, node, ngspice_id):
        voltage[0] = self.f_of_t_and_T(time, self.last_th)
        return 0
    # TODO: Leads to bugs
    def get_isrc_data(self, current, time, node, ngspice_id):
        current[0] = self.f_of_t_and_T(time, self.last_th)
        return 0

class DetectorCircuit(Circuit):
    def __init__(self, name, f_of_t_and_T, voltage_src = True):
        Circuit.__init__(self, name)
        self.f_of_t_and_T = f_of_t_and_T
        ### HEAT SINK
        self.V('1', '3', self.gnd, TAMB@u_V)
        self.R('1', '4', '3', K_RAD@u_Ohm)
        self.C('1', '4', self.gnd, C_RAD@u_F, initial_condition = TAMB@u_V)
        self.R('2', '4', '1', K_SIL@u_Ohm)
        ### THERMAL PELTIER MODEL
        self.C('2', '1', '0', C_H@u_F, initial_condition = TAMB@u_V)
        self.BehavioralSource('1', self.gnd, '1', \
                i = '((v(13) - v(12))/{})*(((v(13) - v(12))/{})*{}+{}*(v(1)-v(2)))'.format(RP, RP, RP, SE))
        self.R('3', '1', '2', K_M@u_Ohm)
        self.BehavioralSource('2', '2', '1', \
                i = '((v(13) - v(12))/{})*({}*v(2)-0.9*((v(13) - v(12))/{}))'.format(RP, SE, RP))
        self.C('3', '2', self.gnd, C_C@u_F, initial_condition = TAMB@u_V)
        ### THERMAL MASS
        self.R('4', '5', '2', K_SIL@u_Ohm)
        self.C('4', '5', self.gnd, C_CONINT@u_F, initial_condition = TAMB@u_V)
        self.R('5', '5', '3', K_CONINT@u_Ohm)
        ### ELECTRICAL PELTIER MODEL
        self.V('2', '11', '13', 0.00@u_V)
        self.R('6', '13', '12', RP@u_Ohm)
        self.VCVS('1', '12', self.gnd, '1', '2', voltage_gain = SE)
        ### EXTERNAL SOURCE
        self.ncs = NgspiceCustomSrc(self.f_of_t_and_T, send_data = True)
        if voltage_src:
            self.V('3', self.gnd, '11', 'dc 0 external')
        else:
            self.I('3', self.gnd, '11', 'dc 0 external')
    def _simulator(self):
        return self.simulator(simulator = 'ngspice-shared', \
                              ngspice_shared = self.ncs)

def test_control_algo(f_of_t, voltage_src = True):
    det = DetectorCircuit('detector_circuit', f_of_t, voltage_src = voltage_src)
    sim = det._simulator()
    sim.options(reltol = 5e-6)
    analysis = sim.transient(step_time = 1.00@u_s, \
                             end_time = 1800.00@u_s, \
                             use_initial_condition = True)
    figure, ax = plt.subplots(figsize=(20, 10))
    ax.plot(analysis['1'], label = 'Hot Side TEC Temperature')
    ax.plot(analysis['2'], label = 'Cold Side TEC Temperature')
    ax.set_xlabel('Time [sec]')
    ax.set_ylabel('Temperature [K]')
    ax.legend()
    ax.set_title('TEC Temperature Transient Characteristic')
    plt.show()

if __name__ == "__main__":
    test_control_algo(lambda t, T : 4.00@u_V)
