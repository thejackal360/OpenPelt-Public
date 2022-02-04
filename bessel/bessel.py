#!/usr/bin/env python3

# Imports

from abc import ABC, abstractmethod

import os
import numpy as np

import cffi
from enum import Enum
import matplotlib.pyplot as plt
from PySpice.Spice.Netlist import Circuit
from PySpice.Unit import u_V, u_A, u_Ohm, u_F, u_s
import PySpice.Spice.NgSpice.Shared

import torch
from torch import nn
from torch.optim import Adam, SGD

from .neural_nets import MLP

# Simulation Parameters

TEMP_SENSOR_SAMPLES_PER_SEC = 1.00
SIMULATION_TIMESTEPS_PER_SENSOR_SAMPLE = 2.00
SIMULATION_TIME_IN_SEC = 5000.00 # 10000.00
ROUND_DIGITS = 3
ERR_TOL = 0.15

COLD_SIDE_NODE = 5
HOT_SIDE_NODE = 4

INPUT_SRC = 'input_src'

# Detector Circuit Parameters

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

# Auxiliary Functions


def K_to_C(T_in_C):
    return T_in_C - 273.15

# Classes


class Signal(Enum):
    VOLTAGE = 1
    CURRENT = 2


class TECPlate(Enum):
    HOT_SIDE = 1
    COLD_SIDE = 2
    

class IndVar(Enum):
    VOLTAGE = 1
    CURRENT = 2
    TIME = 3


class sequencer(ABC):

    @abstractmethod
    def get_ref(self):
        pass


class circular_buffer_sequencer(sequencer):

    def __init__(self, sequence, ngspice_custom_lib):
        self.sequence = sequence
        self.sequence_idx = 0
        self.ngspice_custom_lib = ngspice_custom_lib

    def get_ref(self):
        if self.ngspice_custom_lib.is_steady_state():
            if self.sequence_idx == len(self.sequence) - 1:
                self.sequence_idx = 0
            else:
                self.sequence_idx += 1
        self.ngspice_custom_lib.set_ref(self.sequence[self.sequence_idx], \
                                        self.ngspice_custom_lib.get_plate_select())
        return self.sequence[self.sequence_idx]


class controller(ABC):

    def set_seqr(self, seqr):
        self.seqr = seqr

    def controller_f(self, t, sensor_dict):
        self.ref = self.seqr.get_ref()
        return self._controller_f(t, self.ref, sensor_dict)

    @abstractmethod
    def _controller_f(self, t, ref, sensor_dict):
        pass


class neural_controller(controller):

    def __init__(self,
                 seqr,
                 plate_select,
                 input_units=3,
                 hidden_units=5,
                 bias=True,
                 lrate=0.01):
        adam = True
        self.seqr = seqr
        self.plate_select = plate_select
        self.net = MLP(hidden_units=hidden_units, bias=bias)
        if adam:
            self.optimizer = Adam(self.net.parameters(),
                                  lr=lrate,
                                  weight_decay=1e-4)
        else:
            self.optimizer = SGD(self.net.parameters(),
                                 lr=lrate,
                                 weight_decay=1e-4,
                                 momentum=0.9)
        self.criterion = nn.MSELoss()
        self.loss_ = []

    def learn(self, T_hot, T_cold, T_ref):
        x = torch.from_numpy(np.array([T_hot,
                                       T_cold,
                                       T_ref]).astype('f'))
        self.t_ref = torch.tensor([T_ref], requires_grad=True)
        self.z = torch.tensor([T_cold if self.plate_select == TECPlate.COLD_SIDE else T_hot], requires_grad=True)

        self.optimizer.zero_grad()
        yc = self.net(x)
        # print(self.z.shape, self.t_ref.shape)
        print("T_ref = {}, self.t_ref = {}, T_hot = {}, T_cold = {}".format(T_ref, self.t_ref, T_hot, T_cold))
        loss = self.criterion(self.z, self.t_ref)
        # loss = self.criterion(yc, self.v)
        loss.backward()
        self.optimizer.step()

        # print("Target: %f, Pred: %f" % (self.v.item(), yc.item()))
        # yc = yc.detach().cpu().numpy()[-1] @ u_V
        self.loss_.append(loss.item())
        print("yc = {}".format(yc))
        return yc

    def _controller_f(self, t, ref, sensor_dict):
        yc = self.learn(sensor_dict["th"][-1], sensor_dict["tc"][-1], self.ref)
        return yc.detach().cpu().numpy()[-1] @ u_V
        # x = torch.from_numpy(np.array([sensor_dict["th"],
        #                                sensor_dict["tc"],
        #                                ref]).astype('f'))
        # yc = self.net(x)
        # return np.round(yc.detach().cpu().numpy()[-1], 2)[0]

class bang_bang_controller(controller):

    def __init__(self, seqr):
        self.seqr = seqr

    def _controller_f(self, t, ref, sensor_dict):
        if sensor_dict["th"][-1] < ref:
            return 14.00
        else:
            return 0.00


class pid_controller(controller):

    def __init__(self, seqr, kp, ki, kd, plate_select):
        # https://en.wikipedia.org/wiki/PID_controller
        self.seqr = seqr
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.dt = (1.00/(TEMP_SENSOR_SAMPLES_PER_SEC * SIMULATION_TIMESTEPS_PER_SENSOR_SAMPLE))
        self.prev_err = 0
        self.integral = 0
        self.plate_select = plate_select

    def _controller_f(self, t, ref, sensor_dict):
        error = ref - sensor_dict["th" if self.plate_select == TECPlate.HOT_SIDE else "tc"][-1]
        proportional = error
        self.integral = self.integral + (error * self.dt)
        derivative = (error - self.prev_err) / self.dt
        output = (self.kp * proportional) + \
                 (self.ki * self.integral) + \
                 (self.kd * derivative)
        output = min(16.00, max(-6.00, output))
        self.prev_err = error
        return output


class plant_circuit(Circuit):
    def __init__(self, name, controller_f, sig_type=Signal.VOLTAGE, plate_select = TECPlate.HOT_SIDE):
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
        self.plate_select = plate_select
        self.ncs = tec_lib(self.controller_f, send_data=True, plate_select = self.plate_select)
        if sig_type == Signal.VOLTAGE:
            self.V(INPUT_SRC, '11', self.gnd, 'dc 0 external')
        else:
            self.I(INPUT_SRC, self.gnd, '11', 'dc 0 external')

    def set_plate_select(self, plate_select):
        self.plate_select = plate_select
        self.ncs.set_plate_select(self.plate_select)

    def set_controller_f(self, controller_f):
        self.controller_f = controller_f
        self.ncs.set_controller_f(self.controller_f)

    def get_ncs(self):
        return self.ncs

    def clear(self):
        self.ncs.clear()

    def plot_th_tc(self, ivar, plot_driver = True, include_ref = False):
        fig = plt.figure()
        ax = fig.add_subplot()
        if ivar == IndVar.VOLTAGE:
            ivar_vals = self.ncs.get_v_arr()
        elif ivar == IndVar.CURRENT:
            ivar_vals = self.ncs.get_i_arr()
        else:
            ivar_vals = self.ncs.get_t()
        th_leg_0, = ax.plot(ivar_vals,
                            self.ncs.get_th_actual(),
                            '-xk', lw=1.5,
                            label="Hot Side Temp [C]", c="r")
        tc_leg_0, = ax.plot(ivar_vals,
                            self.ncs.get_tc_actual(),
                            '-.xb', lw=1.5,
                            label="Cold Side Temp [C]", c="b")
        if include_ref:
            ref_leg_0, = ax.plot(ivar_vals,
                                 self.ncs.get_ref_arr(),
                                 '*', lw=1.5,
                                 label="Ref Temp [C]", c = 'y')
        if ivar == IndVar.VOLTAGE:
            ax.set_xlabel("Voltage [V]", fontsize=18,
                          weight='bold', color='black')
        elif ivar == IndVar.CURRENT:
            ax.set_xlabel("Current [A]", fontsize=18,
                          weight='bold', color='black')
        else:
            ax.set_xlabel("Time [s]", fontsize=18,
                          weight='bold', color='black')
        ax.set_ylabel("Temperature [C]", fontsize=18,
                      weight='bold', color='black')
        ax.grid()

        if ivar == IndVar.TIME and plot_driver:
            ax1 = ax.twinx()
            if self.sig_type == Signal.VOLTAGE:
                sig_leg, = ax1.plot(self.ncs.get_t(),
                                    self.ncs.get_v_arr(),
                                    '--', lw=1.5, c='g',
                                    label="Driving Voltage")
                ax1.set_ylabel("Voltage [V]", fontsize=18,
                               weight='bold', color='black')
                ax1.yaxis.label.set_color('black')
                ax1.spines["right"].set_edgecolor('black')
                ax1.tick_params(axis='y', colors='black')
            else:
                sig_leg, = ax1.plot(self.ncs.get_t(),
                                    self.ncs.get_i_arr(),
                                    '--', lw=1.5, c='g',
                                    label="Driving Current")
                ax1.set_ylabel("Current [A]", fontsize=18,
                               weight='bold', color='black')
                ax1.yaxis.label.set_color('black')
                ax1.spines["right"].set_edgecolor('black')
                ax1.tick_params(axis='y', colors='black')
            ax1.legend(fontsize=16)

        ax.legend(fontsize=16)

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
        sim.options(reltol=5e-6)
        anls = sim.transient(step_time=(1.00/(TEMP_SENSOR_SAMPLES_PER_SEC *
                                              SIMULATION_TIMESTEPS_PER_SENSOR_SAMPLE))@u_s,
                             end_time=SIMULATION_TIME_IN_SEC@u_s,
                             use_initial_condition=True)
        Th_final = self.get_th_sensor()[-1]
        Tc_final = self.get_tc_sensor()[-1]
        V_final = self.get_v_arr()[-1]
        return V_final, Th_final, Tc_final

    def characterize_plant(self, val_min, val_max, step_size):
        sim = self.simulator(simulator='ngspice-shared',
                             ngspice_shared=self.ncs)
        sim.options(reltol=5e-6)
        if self.sig_type == Signal.VOLTAGE:
            anls = sim.dc(Vinput_src=slice(val_min, val_max, step_size))
        else:
            anls = sim.dc(Iinput_src=slice(val_min, val_max, step_size))

    def is_steady_state(self):
        return self.ncs.is_steady_state()


class tec_lib(PySpice.Spice.NgSpice.Shared.NgSpiceShared):
    def __init__(self, controller_f, ref = 0.00, steady_state_cycles = 1000, plate_select = TECPlate.HOT_SIDE, **kwargs):
        # Temporary workaround:
        # https://github.com/FabriceSalvaire/PySpice/pull/94
        PySpice.Spice.NgSpice.Shared.ffi = cffi.FFI()
        super().__init__(**kwargs)
        self.controller_f = controller_f
        self.ref = ref
        self.ref_arr = []
        self.plate_select = plate_select
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
        self.steady_state_cycles = steady_state_cycles
        self.th_sensor_window = []
        self.th_sensor_error = []
        self.tc_sensor_window = []
        self.tc_sensor_error = []

    def get_ref_arr(self):
        return self.ref_arr

    def get_ref(self):
        return self.ref

    def get_plate_select(self):
        return self.plate_select

    def set_plate_select(self, plate_select):
        self.plate_select = plate_select

    def set_controller_f(self, controller_f):
        self.controller_f = controller_f

    def set_ref(self, ref, plate_select):
        self.ref = ref
        self.plate_select = plate_select
        if self.ref != 0:
            self.th_sensor_error = [abs(x - self.ref) / abs(self.ref) < ERR_TOL for x in self.th_sensor_window]
            self.tc_sensor_error = [abs(x - self.ref) / abs(self.ref) < ERR_TOL for x in self.tc_sensor_window]
        else:
            # TODO
            self.th_sensor_error = [abs(x) < ERR_TOL for x in self.th_sensor_window]
            self.tc_sensor_error = [abs(x) < ERR_TOL for x in self.tc_sensor_window]

    def send_data(self, actual_vector_values, number_of_vectors, ngspice_id):
        print("t : {}, Th : {}, Tc : {}".format(actual_vector_values['time'].real,
                                                round(K_to_C(actual_vector_values['V({})'.format(HOT_SIDE_NODE)].real), ROUND_DIGITS),
                                                round(K_to_C(actual_vector_values['V({})'.format(COLD_SIDE_NODE)].real), ROUND_DIGITS)))
        if self.timestep_counter == SIMULATION_TIMESTEPS_PER_SENSOR_SAMPLE:
            self.th_sensor.append(
                round(K_to_C(actual_vector_values['V({})'.format(HOT_SIDE_NODE)].real), ROUND_DIGITS))
            if len(self.th_sensor_window) == self.steady_state_cycles:
                del self.th_sensor_window[0]
            self.th_sensor_window.append(self.th_sensor[-1])
            if len(self.th_sensor_error) == self.steady_state_cycles:
                del self.th_sensor_error[0]
            if self.ref != 0:
                self.th_sensor_error.append(abs(self.th_sensor[-1] - self.ref) / self.ref < ERR_TOL)
            else:
                # TODO
                self.th_sensor_error.append(abs(self.th_sensor[-1]) < ERR_TOL)
            self.tc_sensor.append(
                round(K_to_C(actual_vector_values['V({})'.format(COLD_SIDE_NODE)].real), ROUND_DIGITS))
            if len(self.tc_sensor_window) == self.steady_state_cycles:
                del self.tc_sensor_window[0]
            self.tc_sensor_window.append(self.tc_sensor[-1])
            if len(self.tc_sensor_error) == self.steady_state_cycles:
                del self.tc_sensor_error[0]
            if self.ref != 0:
                self.tc_sensor_error.append(abs(self.tc_sensor[-1] - self.ref) / self.ref < ERR_TOL)
            else:
                # TODO
                self.tc_sensor_error.append(abs(self.tc_sensor[-1]) < ERR_TOL)
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
                actual_vector_values['time'].real,
                {"th" : self.th_sensor, "tc" : self.tc_sensor})
            self.next_i = self.controller_f(
                actual_vector_values['time'].real,
                {"th" : self.th_sensor, "tc" : self.tc_sensor})
            self.t.append(actual_vector_values['time'].real)
        except KeyError:
            # DC sweep sim
            pass
        self.ref_arr.append(self.ref)
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
        if self.plate_select == TECPlate.HOT_SIDE:
            if len(self.th_sensor_error) < self.steady_state_cycles:
                return False
            else:
                return all(self.th_sensor_error)
        else:
            if len(self.tc_sensor_error) < self.steady_state_cycles:
                return False
            else:
                return all(self.tc_sensor_error)
