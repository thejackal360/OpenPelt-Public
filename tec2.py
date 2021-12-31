#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
from PySpice.Spice.Netlist import Circuit, SubCircuit
from PySpice.Unit import *

DEFAULT_AMBIENT_C = 27.00
DEFAULT_AMBIENT_K = DEFAULT_AMBIENT_C + 273.15

temp_dep_str = lambda s,tc1,T_plus,T_minus : "({} * (1.00 + ({})*({} - {})))".format(s, tc1, T_plus, T_minus)
vcr_str = lambda p,n,r : "((v({}) - v({}))/({}))".format(p, n, r)

class TempControlledResistor(SubCircuit):
    __nodes__ = ('pin_plus', 'pin_minus', 'ctrl_plus', 'ctrl_minus')
    # R_300 - resistance at ctrl_minus temperature
    # alpha - temperature coefficient of resistivity
    def __init__(self, name, R_300, alpha):
        SubCircuit.__init__(self, name, *self.__nodes__)
        self.R_300 = R_300
        self.alpha = alpha
        print(temp_dep_str(R_300, self.alpha, \
                                         'v(ctrl_plus)', 'v(ctrl_minus)'))
        self.BehavioralSource('1', 'pin_plus', 'pin_minus', \
                i = vcr_str('pin_plus', 'pin_minus', \
                            temp_dep_str(R_300, self.alpha, \
                                         'v(ctrl_plus)', 'v(ctrl_minus)')))

class Thermocouple(SubCircuit):
    __nodes__ = ('pin_plus', 'pin_minus', 'T_plus', 'T_minus')
    # Rn -> n-pellet resistance
    # tcn1 -> tc1 n-pellet
    # Rp -> p-pellet resistance
    # tcp1 -> tc1 p-pellet
    # Rc -> conductor resistance
    # tcc1 -> tc1 conductor
    def __init__(self, name, Rn, tcn1, Rp, tcp1, Rc, tcc1):
        SubCircuit.__init__(self, name, *self.__nodes__)
        self.Rn   = Rn
        self.tcn1 = tcn1
        self.Rp   = Rp
        self.tcp1 = tcp1
        self.Rc   = Rc
        self.tcc1 = tcc1
        circuit.subcircuit(TempControlledResistor(self.Rc / 2.00, self.tcc1))
        circuit.X('1', 'Rc_0', 'pin_plus', 2, 'T_plus', 'T_minus')
        circuit.subcircuit(TempControlledResistor(self.Rn, self.tcn1))
        circuit.X('2', 'Rn_0', 2, 3, 'T_plus', 'T_minus')
        circuit.subcircuit(TempControlledResistor(self.Rc, self.tcc1))
        circuit.X('3', 'Rc_1', 3, 4, 'T_plus', 'T_minus')
        circuit.subcircuit(TempControlledResistor(self.Rp, self.tcp1))
        circuit.X('4', 'Rp_0', 4, 5, 'T_plus', 'T_minus')
        circuit.subcircuit(TempControlledResistor(self.Rc / 2.00, self.tcc1))
        circuit.X('5', 'Rc_1', 5, 'pin_minus', 'T_plus', 'T_minus')

### Test Suite ###

def TCRRun(T0, T1, R, Tc1):
    _R = []
    _T = np.linspace(T0, T1, 100)
    for T in _T:
        circuit = Circuit(str(T))
        tcr = TempControlledResistor('testR', R, Tc1)
        circuit.subcircuit(tcr)
        circuit.X('1', 'testR', 'ckt_top', circuit.gnd, 'T_plus', 'T_minus')
        circuit.V('2', 'T_plus', circuit.gnd, T@u_V)
        circuit.V('3', 'T_minus', circuit.gnd, DEFAULT_AMBIENT_K@u_V)
        circuit.I('4', circuit.gnd, 'ckt_top', 1.00@u_A)
        print(circuit)
        simulator = circuit.simulator(temperature = DEFAULT_AMBIENT_C, nominal_temperature = DEFAULT_AMBIENT_C)
        analysis = simulator.operating_point()
        _R.append(float(analysis['ckt_top']) / 1.00)
    plt.plot(_T, _R, color = 'red')
    plt.show()

if __name__ == "__main__":
    # Temp Controlled Resistor Test
    R = 100.00
    Tc1 = 1e-4
    TCRRun(150.00, 450.00, R, Tc1)
