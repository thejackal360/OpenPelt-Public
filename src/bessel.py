#!/usr/bin/env python3

### Imports ###

import cffi
from enum import Enum
import matplotlib.pyplot as plt
from PySpice.Spice.Netlist import Circuit, SubCircuit
from PySpice.Unit import u_V, u_A, u_Ohm, u_F, u_s
import PySpice.Spice.NgSpice.Shared
import numpy
from initialize_randomness import *
from neural_controller import *

### Simulation Parameters ###

TEMP_SENSOR_SAMPLES_PER_SEC = 5.00
SIMULATION_TIMESTEPS_PER_SENSOR_SAMPLE = 2.00
SIMULATION_TIME_IN_SEC = 3000.00
ROUND_DIGITS = 3

COLD_SIDE_NODE = 5
HOT_SIDE_NODE = 4

INPUT_SRC = 'input_src'

### Detector Circuit Parameters ###

TAMB = 296.4
RP = 1.8
SE = 0.05292
K_RAD = 0.34
C_RAD = 340.00
K_SIL = 0.143
C_H = 2.00
K_M = 1.768
C_C = 2.00
C_CONINT = 304.00
K_CONINT = 3.1

### Auxiliary Functions ###

def K_to_C(T_in_C):
    return T_in_C - 273.15

### Classes ###

class Signal(Enum):
    VOLTAGE = 1
    CURRENT = 2

class IndVar(Enum):
    VOLTAGE = 1
    CURRENT = 2
    TIME    = 3

class PlantCircuit(Circuit):
    def __init__(self, name, controller_f, sig_type=Signal.VOLTAGE):
        Circuit.__init__(self, name)
        self.controller_f = controller_f
        self.sig_type = sig_type
        # HEAT SINK
        self.V('1', '3', self.gnd, TAMB @ u_V)
        self.R('1', '4', '3', K_RAD @ u_Ohm)
        self.C('1', '4', self.gnd, C_RAD@u_F, initial_condition=TAMB@u_V)
        self.R('2', '4', '1', K_SIL@u_Ohm)
        # THERMAL PELTIER MODEL
        self.C('2', '1', '0', C_H@u_F, initial_condition=TAMB@u_V)
        self.BehavioralSource('1', self.gnd, '1',
                              i='((v(13) - v(12))/{})*(((v(13) - v(12))/{})*{}+{}*(v(1)-v(2)))'.format(RP, RP, RP, SE))
        self.R('3', '1', '2', K_M@u_Ohm)
        self.BehavioralSource('2', '2', '1',
                              i='((v(13) - v(12))/{})*({}*v(2)-0.9*((v(13) - v(12))/{}))'.format(RP, SE, RP))
        self.C('3', '2', self.gnd, C_C@u_F, initial_condition=TAMB@u_V)
        # THERMAL MASS
        self.R('4', '5', '2', K_SIL@u_Ohm)
        self.C('4', '5', self.gnd, C_CONINT@u_F, initial_condition=TAMB@u_V)
        self.R('5', '5', '3', K_CONINT@u_Ohm)
        # ELECTRICAL PELTIER MODEL
        self.V('2', '11', '13', 0.00@u_V)
        self.R('6', '13', '12', RP@u_Ohm)
        self.VCVS('1', '12', self.gnd, '1', '2', voltage_gain=SE)
        # EXTERNAL SOURCE
        self.ncs = NgspiceCustomSrc(self.controller_f, send_data=True)
        if sig_type == Signal.VOLTAGE:
            self.V(INPUT_SRC, '11', self.gnd, 'dc 0 external')
        else:
            self.I(INPUT_SRC, self.gnd, '11', 'dc 0 external')

    def clear(self):
        self.ncs.clear()

    def plot_th_tc(self, ivar):
        fig = plt.figure()
        ax = fig.add_subplot()
        if ivar == IndVar.VOLTAGE:
            ivar_vals = self.ncs.get_v_arr()
        elif ivar == IndVar.CURRENT:
            ivar_vals = self.ncs.get_i_arr()
        else:
            ivar_vals = self.ncs.get_t()
        th_leg_0, = ax.plot(ivar_vals, \
                            self.ncs.get_th_actual(), \
                            '-xk', lw = 1.5, \
                            label = "Hot Side Temp [C]", c = "r")
        tc_leg_0, = ax.plot(ivar_vals, \
                            self.ncs.get_tc_actual(), \
                            '-.xb', lw = 1.5, \
                            label = "Cold Side Temp [C]", c = "b")
        if ivar == IndVar.VOLTAGE:
            ax.set_xlabel("Voltage [V]", fontsize = 18, weight = 'bold', color = 'black')
        elif ivar == IndVar.CURRENT:
            ax.set_xlabel("Current [A]", fontsize = 18, weight = 'bold', color = 'black')
        else:
            ax.set_xlabel("Time [s]", fontsize = 18, weight = 'bold', color = 'black')
        ax.set_ylabel("Temperature [C]", fontsize = 18, weight = 'bold', color = 'black')
        ax.grid()

        if ivar == IndVar.TIME:
            ax1 = ax.twinx()
            if self.sig_type == Signal.VOLTAGE:
                sig_leg, = ax1.plot(self.ncs.get_t(),
                                    self.ncs.get_v_arr(),
                                    '--', lw = 1.5, c='g',
                                    label="Driving Voltage")
                ax1.set_ylabel("Voltage [V]", fontsize = 18, weight = 'bold', color = 'black')
                ax1.yaxis.label.set_color('black')
                ax1.spines["right"].set_edgecolor('black')
                ax1.tick_params(axis='y', colors='black')
            else:
                sig_leg, = ax1.plot(self.ncs.get_t(),
                                    self.ncs.get_i_arr(),
                                    '--', lw = 1.5, c='g',
                                    label="Driving Current")
                ax1.set_ylabel("Current [A]", fontsize = 18, weight = 'bold', color = 'black')
                ax1.yaxis.label.set_color('black')
                ax1.spines["right"].set_edgecolor('black')
                ax1.tick_params(axis='y', colors='black')

        ax.legend(fontsize = 16)
        ax1.legend(fontsize = 16)

    def get_t(self):
        return self.ncs.get_t()

    def get_th_actual(self):
        return self.ncs.get_th_actual()

    def get_tc_actual(self):
        return self.ncs.get_tc_actual()

    def get_th_sensor(self):
        return self.ncs.get_th_sensor()

    def get_tc_sensor(self):
        return self.ncs.get_tc_sensor()

    def get_v_arr(self):
        return self.ncs.get_v_arr()

    def get_i_arr(self):
        return self.ncs.get_i_arr()

    def _simulator(self):
        return self.simulator(simulator='ngspice-shared',
                              ngspice_shared=self.ncs)

    def run_sim(self):
        sim = self._simulator()
        sim.options(reltol = 5e-6)
        anls = sim.transient(step_time = \
                  (1.00/(TEMP_SENSOR_SAMPLES_PER_SEC *
                   SIMULATION_TIMESTEPS_PER_SENSOR_SAMPLE))@u_s, \
                             end_time = SIMULATION_TIME_IN_SEC@u_s, \
                             use_initial_condition = True)
        Th_final = self.get_th_sensor()[-1]
        Tc_final = self.get_tc_sensor()[-1]
        V_final  = self.get_v_arr()[-1]
        return V_final, Th_final, Tc_final

    def characterize_plant(self, val_min, val_max, step_size):
        sim = self.simulator(simulator = 'ngspice-shared', \
                             ngspice_shared = self.ncs)
        sim.options(reltol = 5e-6)
        if self.sig_type == Signal.VOLTAGE:
            anls = sim.dc(Vinput_src=slice(val_min, val_max, step_size))
        else:
            anls = sim.dc(Iinput_src=slice(val_min, val_max, step_size))

    def is_steady_state(self):
        return self.ncs.is_steady_state()

class NgspiceCustomSrc(PySpice.Spice.NgSpice.Shared.NgSpiceShared):
    def __init__(self, controller_f, **kwargs):
        # Temporary workaround:
        # https://github.com/FabriceSalvaire/PySpice/pull/94
        PySpice.Spice.NgSpice.Shared.ffi = cffi.FFI()
        super().__init__(**kwargs)
        self.controller_f = controller_f
        self.t = []
        self.th_actual = []
        self.tc_actual = []
        self.th_sensor = []
        self.tc_sensor = []
        self.v = []
        self.i = []
        self.timestep_counter = SIMULATION_TIMESTEPS_PER_SENSOR_SAMPLE
        self.next_v = 0.00
        self.next_i = 0.00

    def send_data(self, actual_vector_values, number_of_vectors, ngspice_id):
        if self.timestep_counter == SIMULATION_TIMESTEPS_PER_SENSOR_SAMPLE:
            self.th_sensor.append(
                round(K_to_C(actual_vector_values['V({})'.format(HOT_SIDE_NODE)].real), ROUND_DIGITS))
            self.tc_sensor.append(
                round(K_to_C(actual_vector_values['V({})'.format(COLD_SIDE_NODE)].real), ROUND_DIGITS))
            self.timestep_counter = 0
        else:
            self.th_sensor.append(self.th_sensor[len(self.th_sensor) - 1])
            self.tc_sensor.append(self.tc_sensor[len(self.tc_sensor) - 1])
            self.timestep_counter += 1
        self.th_actual.append(
            round(K_to_C(actual_vector_values['V({})'.format(HOT_SIDE_NODE)].real), ROUND_DIGITS))
        self.tc_actual.append(
            round(K_to_C(actual_vector_values['V({})'.format(COLD_SIDE_NODE)].real), ROUND_DIGITS))
        self.v.append(actual_vector_values['V(11)'].real)
        self.i.append(
            (actual_vector_values['V(13)'].real - actual_vector_values['V(12)'].real)/RP)
        try:
            self.next_v = self.controller_f(
                actual_vector_values['time'].real, self.th_sensor, self.tc_sensor)
            self.next_i = self.controller_f(
                actual_vector_values['time'].real, self.th_sensor, self.tc_sensor)
            self.t.append(actual_vector_values['time'].real)
        except KeyError:
            # DC sweep sim
            pass
        return 0

    def clear(self):
        self.t = []
        self.th_actual = []
        self.tc_actual = []
        self.th_sensor = []
        self.tc_sensor = []
        self.v = []
        self.i = []
        self.timestep_counter = SIMULATION_TIMESTEPS_PER_SENSOR_SAMPLE
        self.next_v = 0.00
        self.next_i = 0.00

    def get_t(self):
        return self.t

    def get_th_actual(self):
        return self.th_actual

    def get_tc_actual(self):
        return self.tc_actual

    def get_th_sensor(self):
        return self.th_sensor

    def get_tc_sensor(self):
        return self.tc_sensor

    def get_v_arr(self):
        return self.v

    def get_i_arr(self):
        return self.i

    def get_vsrc_data(self, voltage, time, node, ngspice_id):
        voltage[0] = self.next_v
        return 0

    def get_isrc_data(self, current, time, node, ngspice_id):
        current[0] = self.next_i
        return 0

    def is_steady_state(self):
        return len(set(self.th_sensor[len(self.th_sensor) - 15:])) == 1 and \
               len(set(self.tc_sensor[len(self.tc_sensor) - 15:])) == 1

### Controllers ###

def fig11_repro_test(t, Th_arr, Tc_arr):
    return 2.1@u_A

if __name__ == "__main__":

    # Sim Parameters

    plot_not_save = False

    fig11_repro = True
    char_i_repro = True
    char_v_repro = True

    # Initialization

    seed_everything(7777)
    target_temp_in_C = -5.00
    volt_ref, amp_ref = 0.0, 10.0
    nc = neural_controller(T_ref        = target_temp_in_C, \
                           hidden_units = 5, \
                           bias         = False, \
                           lrate        = 1e-3)

    def p_controller(t, Th_arr, Tc_arr):
        return min(12.00,
                   0.5 * (Tc_arr[len(Tc_arr) - 1] - target_temp_in_C)) @ u_V

    def volt_input(t, Th_arr, Tc_arr):
        return volt_ref @ u_V

    def amp_input(t, Th_arr, Tc_arr):
        return amp_ref @ u_A
    
    if fig11_repro:
        pC = PlantCircuit("Detector", fig11_repro_test, Signal.CURRENT)
        pC.run_sim()
        if plot_not_save:
            pC.plot_th_tc(IndVar.TIME)
        else:
            numpy.savez("Fig11ReproductionTh", x = pC.get_t(), y = pC.get_th_sensor())
            numpy.savez("Fig11ReproductionTc", x = pC.get_t(), y = pC.get_tc_sensor())
        if not pC.is_steady_state():
            print("Need sim to run for longer!")
            assert pC.is_steady_state()
        plt.show()

    if char_i_repro:
        pC = PlantCircuit("Detector", fig11_repro_test, Signal.CURRENT)
        pC.characterize_plant(-6.00@u_A, 6.00@u_A, 0.01@u_A)
        if plot_not_save:
            pC.plot_th_tc(IndVar.CURRENT)
        else:
            numpy.savez("ICharacteristicTh", x = pC.get_i_arr(), y = pC.get_th_actual())
            numpy.savez("ICharacteristicTc", x = pC.get_i_arr(), y = pC.get_tc_actual())
        plt.show()

    if char_v_repro:
        pC = PlantCircuit("Detector", fig11_repro_test, Signal.VOLTAGE)
        pC.characterize_plant(-6.00@u_V, 16.4@u_V, 0.01@u_V)
        if plot_not_save:
            pC.plot_th_tc(IndVar.VOLTAGE)
        else:
            numpy.savez("VCharacteristicTh", x = pC.get_v_arr(), y = pC.get_th_actual())
            numpy.savez("VCharacteristicTc", x = pC.get_v_arr(), y = pC.get_tc_actual())
        plt.show()
