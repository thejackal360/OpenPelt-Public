#!/usr/bin/env python3

from PySpice.Spice.Netlist import Circuit, SubCircuit
from PySpice.Unit import *

circuit = Circuit('Metal Cell')

class MetalCell(SubCircuit):
    __nodes__ = ('pin_left', \
                 'pin_top', \
                 'pin_right', \
                 'pin_bottom', \
                 'ambient', \
                 'driver')
    # k -> plate thermal conductivity [W/m*K]
    # delta_tau -> metal thickness [m]
    # L -> side length of cell [m]
    # _L -> side length of central cell capacitance [m]
    # rho -> mass density of plate [kg/m^3]
    # c -> specific heat capacity of plate [J/(kg*K)]
    # h -> convection coefficient [W/m2*K]
    # addtl_conv -> True changes corresponding resistance to convection from conduction
    #     element 0 -> left side resistance
    #     element 1 -> top resistance
    #     element 2 -> right side resistance
    #     element 3 -> bottom resistance
    # disable_conv -> remove convection resistance if True
    #     element 0 -> top plate
    #     element 1 -> bottom plate
    def __init__(self, name, k, delta_tau, L, _L, rho, c, h, \
            addtl_conv = [False, False, False, False], \
            disable_conv = [False, False]):
        SubCircuit.__init__(self, name, *self.__nodes__)
        self.k         = k
        self.delta_tau = delta_tau
        self.L         = L
        self._L        = _L
        self.rho       = rho
        self.c         = c
        self.h         = h
        for i in range(4):
            self.R(i, 'ctr', self.metal_cell_pin_name(i), \
                             self.get_R(convection = addtl_conv[i]))
        self.R(4, 'ctr', 'top_plate', \
                0 if disable_conv[0] else self.get_R(convection = True))
        self.R(5, 'ctr', 'bottom_plate', \
                0 if disable_conv[0] else self.get_R(convection = True))
        self.R(6, 'ctr', 'driver', 0)
        self.C(0, 'ctr', 'ambient', self.get_C())
    def metal_cell_pin_name(self, i):
        assert i in range(4)
        if i == 0:
            return "pin_left"
        elif i == 1:
            return "pin_top"
        elif i == 2:
            return "pin_right"
        else:
            assert i == 3
            return "pin_bottom"
    def get_R(self, convection = False, face_plate = False):
        if convection:
            if face_plate:
                return (1.00 / (self.h * (self.L ** 2.00)))@u_N
            else:
                return (1.00 / (self.h * self.delta_tau * self._L))@u_N
        else:
            return (1.00 / (2.00 * self.k * self.delta_tau))*\
                    ((self.L / self._L) - 1.00)@u_N
    def get_C(self):
        return ((self._L ** 2.00) * self.delta_tau * self.rho * self.c)@u_F

class MetalPlate(SubCircuit):
    def __init__(self, name, k, delta_tau, W, H, rho, c, h, delta_x, delta_sub_x, ambient_temp_in_C):
        nodes = []
        for i in range(H / delta_x):
            for j in range(W / delta_x):
                nodes.append("driver_{}_{}".format(i, j))
        SubCircuit.__init__(self, name, nodes)
        self.k                 = k
        self.delta_tau         = delta_tau
        self.W                 = W
        self.H                 = H
        self.rho               = rho
        self.c                 = c
        self.h                 = h
        self.delta_x           = delta_x
        self.delta_sub_x       = delta_sub_x
        self.ambient_temp_in_C = ambient_temp_in_C
        for i in range(self.H / self.delta_x):
            for j in range(self.W / self.delta_x):
                self.subcircuit(MetalCell('cell_{}_{}'.format(i, j), \
                                          self.k, self.delta_tau, \
                                          self.delta_x, self.delta_sub_x, \
                                          self.rho, self.c, self.h, \
                                          addtl_conv = [True if j == 0 else False, \
                                                        True if i == 0 else False, \
                                                        True if j == (self.W / self.delta_x) - 1 else False, \
                                                        True if i == (self.H / self.delta_x) - 1 else False], \
                                          disable_conv = [False, False]))
                self.X('{}_{}'.format(i, j), 'cell_{}_{}'.format(i, j), \
                        'pin_left_{}_{}'.format(i, j), \
                        'pin_top_{}_{}'.format(i, j), \
                        'pin_right_{}_{}'.format(i, j), \
                        'pin_bottom_{}_{}'.format(i, j), \
                        'ambient', 'driver_{}_{}'.format(i, j))
        for i in range(self.H / self.delta_x):
            for j in range(self.W / self.delta_x):
                self.R('left_right_{}_{}'.format(i, j), \
                       'pin_right_{}_{}'.format(i, j-1) if j != 0 else 'ambient', \
                       'pin_left_{}_{}'.format(i, j), 0)
                self.R('right_left_{}_{}'.format(i, j), \
                       'pin_left_{}_{}'.format(i, j+1) if j != (self.W / self.delta_x) - 1 else 'ambient', \
                       'pin_right_{}_{}'.format(i, j), 0)
                self.R('left_right_{}_{}'.format(i, j), \
                       'pin_bottom_{}_{}'.format(i-1, j) if i != 0 else 'ambient', \
                       'pin_top_{}_{}'.format(i, j), 0)
                self.R('left_right_{}_{}'.format(i, j), \
                       'pin_top_{}_{}'.format(i+1, j) if i != (self.H / self.delta_x) - 1 else 'ambient', \
                       'pin_bottom_{}_{}'.format(i, j), 0)
        self.V('ambient_temp_src', 'ambient', circuit.gnd, ambient_temp_in_C@u_V)

if __name__ == "__main__":
    circuit = Circuit('TestCkt')
    # Aluminum Plate
    ambient_temp = 27.00
    circuit.subcircuit(MetalCell('cell0', 150.00, 6.35e-3, 1e-3, 0.5e-3, 2710.00, 890.00, 12.12))
    circuit.X('1', 'cell0', 1, 2, 3, 4, 5, 6)
    circuit.V(5, 'ambient', circuit.gnd, ambient_temp@u_V)
