#!/usr/bin/env python3

# Imports

from abc import ABC, abstractmethod

import os
import random

import cffi
from enum import Enum
import matplotlib.pyplot as plt
from PySpice.Spice.Netlist import Circuit
from PySpice.Unit import u_V, u_A, u_Ohm, u_F, u_s
# import PySpice.Spice.NgSpice.Shared
import PySpice.Spice.OpenSPICE.Shared
import OpenSPICE

try:
    from fenics import Constant, SubDomain, near
    INCLUDE_FENICS = True
except ImportError:
    INCLUDE_FENICS = False
    print("Warning: Cannot import fenics")

import numpy as np

try:
    import torch
    INCLUDE_TORCH = True
except ImportError:
    INCLUDE_TORCH = False
    print("Warning: cannot import torch")

# Simulation Parameters

DEFAULT_TEMP_SENSOR_SAMPLES_PER_SEC = 1.00
DEFAULT_SIMULATION_TIMESTEPS_PER_SENSOR_SAMPLE = 2.00
DEFAULT_SIMULATION_TIME_IN_SEC = 1500.00
ROUND_DIGITS = 1
ERR_TOL = 0.01

FENICS_TOL = 1e-14
CERAMIC_K = 3.8

COLD_SIDE_NODE = 5
HOT_SIDE_NODE = 4

INPUT_SRC = 'input_src'

# RANDOM SEED
DEFAULT_SEED = 7777

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

def K_to_C(T_in_K):
    """
    Convert temperature in kelvin to temperature in celsius.
    """
    return T_in_K - 273.15


def C_to_K(T_in_C):
    """
    Convert temperature in celsius to temperature in kelvin.
    """
    return T_in_C + 273.15


def seed_everything(seed=1234):
    """
    Set random seed.
    """
    random.seed(seed)
    tseed = random.randint(1, 1E6)
    tcseed = random.randint(1, 1E6)
    npseed = random.randint(1, 1E6)
    ospyseed = random.randint(1, 1E6)
    if INCLUDE_TORCH:
        torch.manual_seed(tseed)
        torch.cuda.manual_seed_all(tcseed)
    np.random.seed(npseed)
    os.environ['PYTHONHASHSEED'] = str(ospyseed)


# Classes
class Signal(Enum):
    """
    Used to select between voltage or current signal type.
    """
    VOLTAGE = 1
    CURRENT = 2


class TECPlate(Enum):
    """
    Used to select hot or cold plate on TEC.
    """
    HOT_SIDE = 1
    COLD_SIDE = 2


class IndVar(Enum):
    """
    Used to select independent variable for SPICE tests
    """
    VOLTAGE = 1
    CURRENT = 2
    TIME = 3


class sequencer(ABC):
    """
    Abstract class for generating sequences of reference values to test
    controllers.
    """

    @abstractmethod
    def get_ref(self):
        """
        Abstract method intended to return next reference value in the sequence
        """
        pass


class circular_buffer_sequencer(sequencer):
    """
    Circular buffer sequence that inherits from abstract sequencer class.
    """

    def __init__(self, sequence, ngspice_custom_lib):
        """
        Initialize sequencer. Need to have some connection to
        ngspice_custom_lib since that (a) specifies the TEC plate in use and
        (b) we need to set the reference there.

        ngspice_custom_lib is of the class type tec_lib.
        """
        self.sequence = sequence
        self.sequence_idx = 0
        self.ngspice_custom_lib = ngspice_custom_lib
        self.ngspice_custom_lib.set_ref(self.sequence[self.sequence_idx],
                                        self.ngspice_custom_lib.get_plate_select())

    def get_ref(self):
        """
        Get the reference value from the circular buffer sequence. Also sets
        the reference value in self.ngspice_custom_lib, intended to be of
        the class type tec_lib.
        """
        if self.ngspice_custom_lib.is_steady_state():
            if self.sequence_idx == len(self.sequence) - 1:
                self.sequence_idx = 0
            else:
                self.sequence_idx += 1
            self.ngspice_custom_lib.set_ref(self.sequence[self.sequence_idx],
                                            self.ngspice_custom_lib.get_plate_select())
        return self.sequence[self.sequence_idx]


if INCLUDE_FENICS:
    class BottomBoundary(SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and near(x[1], -0.002, FENICS_TOL)

class op_amp_lib(PySpice.Spice.OpenSPICE.Shared.OpenSPICEShared):

    def __init__(self,
                 controller_f,
                 sim_timesteps_per_sensor_sample=DEFAULT_SIMULATION_TIMESTEPS_PER_SENSOR_SAMPLE,
                 **kwargs):
        # Temporary workaround:
        # https://github.com/FabriceSalvaire/PySpice/pull/94
        PySpice.Spice.OpenSPICE.Shared.ffi = cffi.FFI()
        super().__init__(**kwargs)
        self.controller_f = controller_f
        self.t = []
        self.v = []
        self.v_in = []
        self.sim_timesteps_per_sensor_sample = sim_timesteps_per_sensor_sample
        self.timestep_counter = self.sim_timesteps_per_sensor_sample
        self.next_v = 0.00
        OpenSPICE.set_get_vsrc(self.get_vsrc_data)
        OpenSPICE.set_send_data(self.send_data)

    def set_controller_f(self, controller_f):
        self.controller_f = controller_f

    def send_data(self, actual_vector_values, number_of_vectors, ngspice_id):
        self.v.append(actual_vector_values['V(3)'].real)
        self.v_in.append(actual_vector_values['V(1)'].real)
        self.next_v = self.controller_f()
        self.t.append(actual_vector_values['time'].real)
        return 0

    def clear(self):
        self.t = []
        self.v = []
        self.timestep_counter = self.sim_timesteps_per_sensor_sample
        self.next_v = 0.00
        self.next_i = 0.00

    def get_t(self):
        return self.t

    def get_v(self):
        return self.v

    def get_v_in(self):
        return self.v_in

    def get_vsrc_data(self, voltage, time, node, ngspice_id):
        voltage[0] = self.next_v
        return 0

class op_amp_plant(Circuit):

    def __init__(self,
                 name,
                 controller_f,
                 steady_state_cycles=1000,
                 sim_time_in_s=DEFAULT_SIMULATION_TIME_IN_SEC,
                 sim_timesteps_per_sensor_sample=DEFAULT_SIMULATION_TIMESTEPS_PER_SENSOR_SAMPLE,
                 temp_sensor_samples_per_s=DEFAULT_TEMP_SENSOR_SAMPLES_PER_SEC):
        Circuit.__init__(self, name)
        self.controller_f = controller_f
        self.sim_time_in_s = sim_time_in_s
        self.sim_timesteps_per_sensor_sample = sim_timesteps_per_sensor_sample
        self.temp_sensor_samples_per_s = temp_sensor_samples_per_s
        self.V(INPUT_SRC, '1', self.gnd, 'dc 0 external')
        self.R('in', '1', self.gnd, 1e6@u_Ohm)
        self.VCVS('out', '3', self.gnd, '1', self.gnd, voltage_gain=1e6)
        self.R('out', '3', self.gnd, 1e6@u_Ohm)
        self.ncs = op_amp_lib(self.controller_f,
                              send_data=True)

    def set_controller_f(self, controller_f):
        self.controller_f = controller_f
        self.ncs.set_controller_f(self.controller_f)

    def get_ncs(self):
        return self.ncs

    def clear(self):
        self.ncs.clear()

    def plot_output_v(self):
        fig = plt.figure()
        ax = fig.add_subplot()
        ivar_vals = self.ncs.get_t()
        v_leg_0, = ax.plot(ivar_vals,
                           self.ncs.get_v(),
                           '-k', lw=1.5,
                           label="Output Voltage [V]", c="r")
        in_v_leg_0, = ax.plot(ivar_vals,
                              self.ncs.get_v_in(),
                              '-k', lw=1.5,
                              label="Input Voltage [V]", c="g")
        ax.set_xlabel("Time [s]", fontsize=18,
                      weight='bold', color='black')
        ax.set_ylabel("Voltage [V]", fontsize=18,
                      weight='bold', color='black')
        ax.grid()
        ax.legend(fontsize=16)

    def get_t(self):
        return self.ncs.get_t()

    def get_v(self):
        return self.ncs.get_v()

    def _simulator(self):
        return self.simulator(simulator='OpenSPICE',
                              ngspice_shared=self.ncs)

    def run_sim(self):
        sim = self._simulator()
        sim.options(reltol=5e-6)
        anls = sim.transient(step_time=(1.00/(self.temp_sensor_samples_per_s *
                                              self.sim_timesteps_per_sensor_sample))@u_s,
                             end_time=self.sim_time_in_s@u_s,
                             use_initial_condition=True)

class rc_ckt_plant(Circuit):

    def __init__(self,
                 name,
                 controller_f,
                 steady_state_cycles=1000,
                 res=1.00,
                 cap=1.00,
                 sim_time_in_s=DEFAULT_SIMULATION_TIME_IN_SEC,
                 sim_timesteps_per_sensor_sample=DEFAULT_SIMULATION_TIMESTEPS_PER_SENSOR_SAMPLE,
                 temp_sensor_samples_per_s=DEFAULT_TEMP_SENSOR_SAMPLES_PER_SEC):
        Circuit.__init__(self, name)
        self.controller_f = controller_f
        self.sim_time_in_s = sim_time_in_s
        self.sim_timesteps_per_sensor_sample = sim_timesteps_per_sensor_sample
        self.temp_sensor_samples_per_s = temp_sensor_samples_per_s
        self.R('1', '1', '2', res @ u_Ohm)
        self.C('1', '2', self.gnd, cap @ u_F, initial_condition=0.00)
        self.ncs = rc_ckt_lib(self.controller_f,
                              send_data=True)
        self.V(INPUT_SRC, '1', self.gnd, 'dc 0 external')

    def set_controller_f(self, controller_f):
        self.controller_f = controller_f
        self.ncs.set_controller_f(self.controller_f)

    def get_ncs(self):
        return self.ncs

    def clear(self):
        self.ncs.clear()

    def plot_output_v(self):
        fig = plt.figure()
        ax = fig.add_subplot()
        ivar_vals = self.ncs.get_t()
        v_leg_0, = ax.plot(ivar_vals,
                           self.ncs.get_v(),
                           '-k', lw=1.5,
                           label="Capacitor Voltage [V]", c="r")
        ax.set_xlabel("Time [s]", fontsize=18,
                      weight='bold', color='black')
        ax.set_ylabel("Voltage [V]", fontsize=18,
                      weight='bold', color='black')
        ax.grid()
        ax.legend(fontsize=16)

    def get_t(self):
        return self.ncs.get_t()

    def get_v(self):
        return self.ncs.get_v()

    def _simulator(self):
        return self.simulator(simulator='OpenSPICE',
                              ngspice_shared=self.ncs)

    def run_sim(self):
        sim = self._simulator()
        sim.options(reltol=5e-6)
        anls = sim.transient(step_time=(1.00/(self.temp_sensor_samples_per_s *
                                              self.sim_timesteps_per_sensor_sample))@u_s,
                             end_time=self.sim_time_in_s@u_s,
                             use_initial_condition=True)

class rc_ckt_lib(PySpice.Spice.OpenSPICE.Shared.OpenSPICEShared):

    def __init__(self,
                 controller_f,
                 plate_select=TECPlate.HOT_SIDE,
                 sim_timesteps_per_sensor_sample=DEFAULT_SIMULATION_TIMESTEPS_PER_SENSOR_SAMPLE,
                 **kwargs):
        # Temporary workaround:
        # https://github.com/FabriceSalvaire/PySpice/pull/94
        PySpice.Spice.OpenSPICE.Shared.ffi = cffi.FFI()
        super().__init__(**kwargs)
        self.controller_f = controller_f
        self.t = []
        self.v = []
        self.sim_timesteps_per_sensor_sample = sim_timesteps_per_sensor_sample
        self.timestep_counter = self.sim_timesteps_per_sensor_sample
        self.next_v = 0.00
        OpenSPICE.set_get_vsrc(self.get_vsrc_data)
        OpenSPICE.set_send_data(self.send_data)

    def set_controller_f(self, controller_f):
        self.controller_f = controller_f

    def send_data(self, actual_vector_values, number_of_vectors, ngspice_id):
        self.v.append(actual_vector_values['V(2)'].real)
        self.next_v = self.controller_f()
        self.t.append(actual_vector_values['time'].real)
        return 0

    def clear(self):
        self.t = []
        self.v = []
        self.timestep_counter = self.sim_timesteps_per_sensor_sample
        self.next_v = 0.00
        self.next_i = 0.00

    def get_t(self):
        return self.t

    def get_v(self):
        return self.v

    def get_vsrc_data(self, voltage, time, node, ngspice_id):
        voltage[0] = self.next_v
        return 0

class tec_plant(Circuit):
    """
    tec_plant class, inherits from PySpice's Circuit object. Meant to be
    a electro-thermal circuit model of a TEC-based mid-IR detector cooler.
    """
    def __init__(self,
                 name,
                 controller_f,
                 sig_type=Signal.VOLTAGE,
                 plate_select=TECPlate.HOT_SIDE,
                 steady_state_cycles=1000,
                 _k_rad=K_RAD,
                 _c_rad=C_RAD,
                 _k_sil=K_SIL,
                 _c_h=C_H,
                 _c_c=C_C,
                 _k_m=K_M,
                 _c_conint=C_CONINT,
                 _k_conint=K_CONINT,
                 _rp=RP,
                 _se=SE,
                 _tamb=TAMB,
                 sim_time_in_s=DEFAULT_SIMULATION_TIME_IN_SEC,
                 sim_timesteps_per_sensor_sample=DEFAULT_SIMULATION_TIMESTEPS_PER_SENSOR_SAMPLE,
                 temp_sensor_samples_per_s=DEFAULT_TEMP_SENSOR_SAMPLES_PER_SEC):
        """
        Instantiate circuit with controller algorithm, whose function is
        specified by controller_f.

        sig_type specifies whether we're driving a voltage or a current from
        the controller.

        The timestep parameters sim_timesteps_per_sensor_sample and temp_sensor_samples_per_s
        are used to determine a suggested timestep size. However, it should be emphasized
        that that timestep size is merely a SUGGESTION to the ngspice simulator. The timestep
        size may vary throughout the simulation, and it depends on other parameters such as
        reltol. Because of this, steady_state_cycles is somewhat ambiguous as well since it
        is given in timesteps.
        """
        Circuit.__init__(self, name)
        self.controller_f = controller_f
        self.sig_type = sig_type
        self.sim_time_in_s = sim_time_in_s
        self.sim_timesteps_per_sensor_sample = sim_timesteps_per_sensor_sample
        self.temp_sensor_samples_per_s = temp_sensor_samples_per_s
        # HEAT SINK
        self.V('1', '3', self.gnd, _tamb @ u_V)
        self.R('1', '4', '3', _k_rad @ u_Ohm)
        self.C('1', '4', self.gnd, _c_rad@u_F, initial_condition=_tamb@u_V)
        self.R('2', '4', '1', _k_sil@u_Ohm)
        # THERMAL PELTIER MODEL
        self.C('2', '1', '0', _c_h@u_F, initial_condition=_tamb@u_V)
        self.BehavioralSource('1',
                              self.gnd,
                              '1',
                              i='((v(13)-v(12))/{})*(((v(13)-v(12))/{})*{}+{}*(v(1)-v(2)))'.format(_rp, _rp, _rp, _se))
        self.R('3', '1', '2', _k_m@u_Ohm)
        self.BehavioralSource('2',
                              '2',
                              '1',
                              i='((v(13)-v(12))/{})*({}*v(2)-0.9*((v(13)-v(12))/{}))'.format(_rp, _se, _rp))
        self.C('3', '2', self.gnd, _c_c@u_F, initial_condition=_tamb@u_V)
        # THERMAL MASS
        self.R('4', '5', '2', _k_sil@u_Ohm)
        self.C('4', '5', self.gnd, _c_conint@u_F, initial_condition=_tamb@u_V)
        self.R('5', '5', '3', _k_conint@u_Ohm)
        # ELECTRICAL PELTIER MODEL
        self.V('2', '11', '13', 0.00@u_V)
        self.R('6', '13', '12', RP@u_Ohm)
        self.VCVS('1', '12', self.gnd, '1', '2', voltage_gain=_se)
        # EXTERNAL SOURCE
        self.plate_select = plate_select
        self.ncs = tec_lib(self.controller_f,
                           send_data=True,
                           plate_select=self.plate_select,
                           steady_state_cycles=steady_state_cycles)
        if sig_type == Signal.VOLTAGE:
            self.V(INPUT_SRC, '11', self.gnd, 'dc 0 external')
        else:
            self.I(INPUT_SRC, self.gnd, '11', 'dc 0 external')
        # Fenics initialization code
        if INCLUDE_FENICS:
            self.subdomain = BottomBoundary()
            self.n = 0
            self.u_D = Constant(K_to_C(_tamb))

    def time_update(self):
        """
        Do not call until after you've run a transient simulation. Coupling to
        the 3D Fenics model is not yet supported.
        """
        th_sensor_arr = self.ncs.get_th_sensor()
        assert self.n < len(th_sensor_arr)
        tc_sensor_arr = self.ncs.get_tc_sensor()
        assert self.n < len(tc_sensor_arr)
        if self.plate_select == TECPlate.HOT_SIDE:
            self.u_D.assign(th_sensor_arr[self.n])
        else:
            self.u_D.assign(tc_sensor_arr[self.n])
        self.n += 1

    def get_k_val(self):
        return CERAMIC_K

    def set_plate_select(self, plate_select):
        """
        Set hot plate or cold plate to be controlled.
        """
        self.plate_select = plate_select
        self.ncs.set_plate_select(self.plate_select)

    def set_controller_f(self, controller_f):
        """
        Set the controller function.
        """
        self.controller_f = controller_f
        self.ncs.set_controller_f(self.controller_f)

    def get_ncs(self):
        """
        Get the underlying tec_lib object that directly interfaces with
        ngspice library using PySpice.
        """
        return self.ncs

    def clear(self):
        """
        Clear tec_lib object.
        """
        self.ncs.clear()

    def plot_th_tc(self, ivar, plot_driver=True, include_ref=False):
        """
        Plot hot side and cold side temperatures of TEC.

        ivar specifies the independent variable (time for transient sim,
        voltage/current for DC sweep sims).

        plot_driver enables plotting the driving voltage/current.

        include_ref enables plotting the reference temperature. Depending on
        how the sequencer is configured, this may change throughout a transient
        sim.
        """
        fig = plt.figure()
        ax = fig.add_subplot()
        if ivar == IndVar.VOLTAGE:
            ivar_vals = self.ncs.get_v_arr()
        elif ivar == IndVar.CURRENT:
            ivar_vals = self.ncs.get_i_arr()
        else:
            ivar_vals = self.ncs.get_t()
        assert len(ivar_vals) == len(self.ncs.get_th_actual())
        assert len(ivar_vals) == len(self.ncs.get_tc_actual())
        if ivar == IndVar.TIME:
            assert len(ivar_vals) == len(self.ncs.get_ref_arr())
            if self.sig_type == Signal.VOLTAGE:
                assert len(ivar_vals) == len(self.ncs.get_v_arr())
            elif self.sig_type == Signal.CURRENT:
                assert len(ivar_vals) == len(self.ncs.get_i_arr())
        th_leg_0, = ax.plot(ivar_vals,
                            self.ncs.get_th_actual(),
                            '-k', lw=1.5,
                            label="Hot Side Temp [C]", c="r")
        tc_leg_0, = ax.plot(ivar_vals,
                            self.ncs.get_tc_actual(),
                            '-b', lw=1.5,
                            label="Cold Side Temp [C]", c="b")
        if include_ref:
            ref_leg_0, = ax.plot(ivar_vals,
                                 self.ncs.get_ref_arr(),
                                 '-', lw=1.5,
                                 label="Ref Temp [C]", c='y')
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
                                    '--', lw=1., c='g',
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
        """
        Get array of timesteps.
        """
        return self.ncs.get_t()

    def get_th_actual(self):
        """
        Get array of hot plate temperatures.
        """
        return self.ncs.get_th_actual()

    def get_tc_actual(self):
        """
        Get array of cold plate temperatures.
        """
        return self.ncs.get_tc_actual()

    def get_th_sensor(self):
        """
        Get array of hot plate temperatures read by sensor. Sensor only reads
        periodically depending on the value of the self.temp_sensor_samples_per_s
        constant.
        """
        return self.ncs.get_th_sensor()

    def get_tc_sensor(self):
        """
        Get array of cold plate temperatures read by sensor. Sensor only reads
        periodically depending on the value of the self.temp_sensor_samples_per_s
        constant.
        """
        return self.ncs.get_tc_sensor()

    def get_v_arr(self):
        """
        Get array of driving voltages. Undefined behavior if current selected.
        """
        return self.ncs.get_v_arr()

    def get_i_arr(self):
        """
        Get array of driving currents. Undefined behavior if voltage selected.
        """
        return self.ncs.get_i_arr()

    def _simulator(self):
        """
        Return simulator object that uses tec_lib ngspice shared library
        object.
        """
        return self.simulator(simulator='OpenSPICE',
                              ngspice_shared=self.ncs)

    def run_sim(self):
        """
        Run a transient simulation. Returns [V_final, Th_final, Tc_final].
        Behavior undefined for current driver simulation.

        Note: Timestep size parameters are merely a suggestion to the SPICE
        simulator. ngspice determines timestep size using numerous factors.
        For instance, reltol for the Newton-Raphson algorithm affects timestep
        size. The timestep size can change throughout the simulation as well.

        Note: reltol is defaulted to 5e-6. This is the reltol in the original paper.
        Future changes may need to adapt this for different scenarios, or simple
        offer it as a parameter to the end user.
        """
        sim = self._simulator()
        sim.options(reltol=5e-6)
        anls = sim.transient(step_time=(1.00/(self.temp_sensor_samples_per_s *
                                              self.sim_timesteps_per_sensor_sample))@u_s,
                             end_time=self.sim_time_in_s@u_s,
                             use_initial_condition=True)
        Th_final = self.get_th_sensor()[-1]
        Tc_final = self.get_tc_sensor()[-1]
        V_final = self.get_v_arr()[-1]
        return V_final, Th_final, Tc_final

    def characterize_plant(self, val_min, val_max, step_size):
        """
        Acts as a DC sweep function. May contain errors since behavior is not
        validated.
        """
        num_incr = (val_max - val_min) / step_size
        assert float(int(num_incr)) == num_incr
        num_incr = int(num_incr)
        sim = self.simulator(simulator='OpenSPICE',
                             ngspice_shared=self.ncs)
        sim.options(reltol=5e-6)
        tmp_controller_f = self.controller_f
        th = []
        tc = []
        if self.sig_type == Signal.VOLTAGE:
            v_range = np.linspace(val_min, val_max, num_incr)
            for _v in v_range:
                print("Testing {}V input...".format(_v))
                self.set_controller_f(lambda _t, _dict : _v)
                tmp_v, tmp_th, tmp_tc = self.run_sim()
                th.append(tmp_th)
                tc.append(tmp_tc)
            self.ncs.v = v_range
        else:
            i_range = np.linspace(val_min, val_max, num_incr)
            for _i in i_range:
                print("Testing {}A input...".format(_i))
                self.set_controller_f(lambda _t, _dict : _i)
                tmp_v, tmp_th, tmp_tc = self.run_sim()
                th.append(tmp_th)
                tc.append(tmp_tc)
            self.ncs.i = i_range
        self.ncs.th_actual = th
        self.ncs.tc_actual = tc
        assert len(th) == len(self.get_th_actual())
        assert len(tc) == len(self.get_tc_actual())
        self.set_controller_f(tmp_controller_f)

    def is_steady_state(self):
        """
        Have we reached steady state? Only useful for running transient
        simulations.
        """
        return self.ncs.is_steady_state()


class tec_lib(PySpice.Spice.OpenSPICE.Shared.OpenSPICEShared):
    """
    Class that inherits from PySpice's OpenSPICEShared class. Abstracts ngspice
    details from the rest of the library.
    """

    def __init__(self,
                 controller_f,
                 ref=0.00,
                 steady_state_cycles=1500,
                 plate_select=TECPlate.HOT_SIDE,
                 sim_timesteps_per_sensor_sample=DEFAULT_SIMULATION_TIMESTEPS_PER_SENSOR_SAMPLE,
                 **kwargs):
        """
        Initialize tec_lib. steady_state_cycles defines the number of
        cycles for which values must be nearly constant before considering
        system to be in steady state.
        """
        # Temporary workaround:
        # https://github.com/FabriceSalvaire/PySpice/pull/94
        PySpice.Spice.OpenSPICE.Shared.ffi = cffi.FFI()
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
        self.sim_timesteps_per_sensor_sample = sim_timesteps_per_sensor_sample
        self.timestep_counter = self.sim_timesteps_per_sensor_sample
        self.next_v = 0.00
        self.next_i = 0.00
        assert steady_state_cycles > 0
        self.steady_state_cycles = steady_state_cycles
        self.th_sensor_error = []
        self.tc_sensor_error = []
        OpenSPICE.set_get_vsrc(self.get_vsrc_data)
        OpenSPICE.set_get_isrc(self.get_isrc_data)
        OpenSPICE.set_send_data(self.send_data)

    def get_ref_arr(self):
        """
        Return array of reference values over time.
        """
        return self.ref_arr

    def get_ref(self):
        """
        Get current reference value.
        """
        return self.ref

    def get_plate_select(self):
        """
        Get the current selected plate for control, either hot side or cold
        side.
        """
        return self.plate_select

    def set_plate_select(self, plate_select):
        """
        Set the plate to be controlled, either hot side or cold side.
        """
        self.plate_select = plate_select

    def set_controller_f(self, controller_f):
        """
        Define the controller function.
        """
        self.controller_f = controller_f

    def set_ref(self, ref, plate_select):
        """
        Set the reference and plate to be controlled.
        """
        self.ref = ref
        self.plate_select = plate_select
        # Need to clear th_sensor_error and tc_sensor_error arrays.
        # Arrays determined by relative error of each reading with the
        # previous reading. Thus, when the reference is changed, the
        # system will still technically be in steady state since the
        # error arrays do not depend on the reference.
        self.th_sensor_error = []
        self.tc_sensor_error = []

    def send_data(self, actual_vector_values, number_of_vectors, ngspice_id):
        """
        Function that determines how to collect data from each timestep of
        ngspice simulation.
        """
        if self.timestep_counter == self.sim_timesteps_per_sensor_sample:
            assert len(self.th_sensor) == len(self.tc_sensor)
            assert len(self.th_sensor_error) == len(self.tc_sensor_error)
            self.th_sensor.append(
                round(K_to_C(actual_vector_values['V({})'.format(HOT_SIDE_NODE)].real), ROUND_DIGITS))
            if len(self.th_sensor_error) == self.steady_state_cycles:
                del self.th_sensor_error[0]
            # Convention: The first element of the error array is always True.
            # All other elements are determined by the relative error between the
            # previous sensor reading and the sensor reading at that index.

            # We're calculating errors using relative error.
            # Relative error only makes sense on ratio scales, like kelvin, as
            # opposed to interval scales, like celsius.
            # https://en.wikipedia.org/wiki/Approximation_error#Examples
            if len(self.th_sensor_error) == 0:
                self.th_sensor_error.append(True)
            else:
                self.th_sensor_error.append(abs(C_to_K(self.th_sensor[-2]) -
                                                C_to_K(self.th_sensor[-1])) /
                                                C_to_K(self.th_sensor[-2]) < ERR_TOL)
            self.tc_sensor.append(
                round(K_to_C(actual_vector_values['V({})'.format(COLD_SIDE_NODE)].real), ROUND_DIGITS))
            if len(self.tc_sensor_error) == self.steady_state_cycles:
                del self.tc_sensor_error[0]
            if len(self.tc_sensor_error) == 0:
                self.tc_sensor_error.append(True)
            else:
                self.tc_sensor_error.append(abs(C_to_K(self.tc_sensor[-2]) -
                                                C_to_K(self.tc_sensor[-1])) /
                                                C_to_K(self.tc_sensor[-2]) < ERR_TOL)
            self.timestep_counter = 0
            assert len(self.th_sensor) == len(self.tc_sensor)
            assert len(self.th_sensor_error) == len(self.tc_sensor_error)
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
        """
        Clear data in all arrays.
        """
        self.t = []
        self.th_actual = []
        self.tc_actual = []
        self.th_sensor = []
        self.tc_sensor = []
        self.v = []
        self.i = []
        self.timestep_counter = self.sim_timesteps_per_sensor_sample
        self.next_v = 0.00
        self.next_i = 0.00

    def get_t(self):
        """
        Get current timestep.
        """
        return self.t

    def get_th_actual(self):
        """
        Get current hot plate temperature.
        """
        return self.th_actual

    def get_tc_actual(self):
        """
        Get current cold plate temperature.
        """
        return self.tc_actual

    def get_th_sensor(self):
        """
        Get current hot plate sensor reading.
        """
        return self.th_sensor

    def get_tc_sensor(self):
        """
        Get current cold plate sensor reading.
        """
        return self.tc_sensor

    def get_v_arr(self):
        """
        Get array of driving voltages throughout sim. Undefined behavior for
        current driving sims.
        """
        return self.v

    def get_i_arr(self):
        """
        Get array of driving currents throughout sim. Undefined behavior for
        voltage driving sims.
        """
        return self.i

    def get_vsrc_data(self, voltage, time, node, ngspice_id):
        """
        Set external voltage source driving value in the ngspice circuit.
        """
        voltage[0] = self.next_v
        return 0

    def get_isrc_data(self, current, time, node, ngspice_id):
        """
        Set external current source driving value in the ngspice circuit.
        """
        current[0] = self.next_i
        return 0

    def is_steady_state(self):
        """
        Check if system has hit steady state.
        """
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
