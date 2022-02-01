#!/usr/bin/env python3
import numpy as np
import PySpice.Spice.NgSpice.Shared
import cffi
from PySpice.Spice.Netlist import Circuit, SubCircuit
from PySpice.Unit import u_V, u_A, u_Ohm, u_F, u_s
import matplotlib.pylab as plt

from neural_controller import neural_controller

from initialize_randomness import seed_everything


TEMP_SENSOR_SAMPLES_PER_SEC = 5.00
SIMULATION_TIMESTEPS_PER_SENSOR_SAMPLE = 2.00
# SIMULATION_TIME_IN_SEC = 1800.00
SIMULATION_TIME_IN_SEC = 3000.00

COLD_SIDE_NODE = 5  # 2
HOT_SIDE_NODE = 4  # 1


def K_to_C(T_in_C):
    return T_in_C - 273.15


# Detector Circuit Parameters #
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
                K_to_C(actual_vector_values['V({})'.format(HOT_SIDE_NODE)].real))
            self.tc_sensor.append(
                K_to_C(actual_vector_values['V({})'.format(COLD_SIDE_NODE)].real))
            self.timestep_counter = 0
        else:
            self.th_sensor.append(self.th_sensor[len(self.th_sensor) - 1])
            self.tc_sensor.append(self.tc_sensor[len(self.tc_sensor) - 1])
            self.timestep_counter += 1
        self.th_actual.append(
            K_to_C(actual_vector_values['V({})'.format(HOT_SIDE_NODE)].real))
        self.tc_actual.append(
            K_to_C(actual_vector_values['V({})'.format(COLD_SIDE_NODE)].real))
        self.v.append(actual_vector_values['V(11)'].real)
        self.i.append(
            (actual_vector_values['V(13)'].real - actual_vector_values['V(12)'].real)/RP)
        self.next_v = self.controller_f(
            actual_vector_values['time'].real, self.th_sensor, self.tc_sensor)
        self.next_i = self.controller_f(
            actual_vector_values['time'].real, self.th_sensor, self.tc_sensor)
        self.t.append(actual_vector_values['time'].real)
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


class DetectorCircuit(Circuit):
    def __init__(self, name, controller_f, voltage_src=True):
        Circuit.__init__(self, name)
        self.controller_f = controller_f
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
        if voltage_src:
            self.V('3', '11', self.gnd, 'dc 0 external')
        else:
            self.I('3', self.gnd, '11', 'dc 0 external')

    def clear(self):
        print("In clear fn: {}".format(len(self.ncs.get_th_actual())))
        # self.ncs.remove_circuit()
        self.ncs.destroy()
        self.ncs.clear()
        print("In clear fn: {}".format(len(self.ncs.get_th_actual())))

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


def test_control_algo(controller_f, voltage_src=True, is_plot_on=False):
    # Constant current - Start
    def constant_current(t, Th_arr, Tc_arr):
        return 2.1 @ u_A if t > 0.00 else 0.0 @ u_A

    det_cnst_current = DetectorCircuit('det_cnst_current_circuit',
                                       constant_current,
                                       voltage_src=False)
    sim_cnst_current = det_cnst_current._simulator()
    sim_cnst_current.options(reltol=5e-6)
    analysis_cnst_current = sim_cnst_current.transient(
        step_time=(1.00/(TEMP_SENSOR_SAMPLES_PER_SEC *
                   SIMULATION_TIMESTEPS_PER_SENSOR_SAMPLE))@u_s,
        end_time=SIMULATION_TIME_IN_SEC@u_s,
        use_initial_condition=True)

    if is_plot_on:
        fig = plt.figure()
        ax = fig.add_subplot()
        th_leg_0, = ax.plot(det_cnst_current.get_t(),
                            det_cnst_current.get_th_actual(),
                            c='r',
                            label="Hot side TEC heat sink temp")
        tc_leg_0, = ax.plot(det_cnst_current.get_t(),
                            det_cnst_current.get_tc_actual(),
                            c='b',
                            label="Cold side TEC heat sink temp")
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Temperature [C]")
        ax1 = ax.twinx()
        curr_leg, = ax1.plot(det_cnst_current.get_t(),
                             det_cnst_current.get_i_arr(),
                             c='g',
                             label="Driving current")
        ax1.set_ylabel("Current [A]")
        ax1.yaxis.label.set_color(curr_leg.get_color())
        ax1.spines["right"].set_edgecolor(curr_leg.get_color())
        ax1.tick_params(axis='y', colors=curr_leg.get_color())
        ax.legend()

    # Constant current - End
    # p controller - Start
    # TODO - detector circuit needs to be instantiated later since it uses
    # the same instance of the ngspice shared library. Fix PySpice issue.
    det_p_ctrl = DetectorCircuit('det_p_ctrl_circuit',
                                 controller_f,
                                 voltage_src=voltage_src)
    sim_p_controller = det_p_ctrl._simulator()
    sim_p_controller.options(reltol=5e-6)
    analysis_p_controller = sim_p_controller.transient(
        step_time=(1.00/(TEMP_SENSOR_SAMPLES_PER_SEC *
                   SIMULATION_TIMESTEPS_PER_SENSOR_SAMPLE))@u_s,
        end_time=SIMULATION_TIME_IN_SEC@u_s,
        use_initial_condition=True)

    if is_plot_on:
        fig, ax = plt.subplots()
        th_leg_1, = ax.plot(det_p_ctrl.get_t(),
                            det_p_ctrl.get_th_actual(),
                            '-r',
                            label="Hot side TEC heat sink temp")
        tc_leg_1, = ax.plot(det_p_ctrl.get_t(),
                            det_p_ctrl.get_tc_actual(),
                            '-b',
                            label="Cold side TEC heat sink temp")
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Temperature [C]")
        ax1 = ax.twinx()

        v_leg, = ax1.plot(det_p_ctrl.get_t(),
                          det_p_ctrl.get_v_arr(),
                          '-g',
                          label="Applied voltage")
        ax1.set_ylabel("Voltage [V]")
        ax.legend()
        ax1.yaxis.label.set_color(v_leg.get_color())
        ax1.spines["right"].set_edgecolor(v_leg.get_color())
        ax1.tick_params(axis='y', colors=v_leg.get_color())

    # p controller - End
    Th_final = det_p_ctrl.get_th_actual()[-1]
    Tc_final = det_p_ctrl.get_tc_actual()[-1]
    volt = det_p_ctrl.get_v_arr()[-1]
    return volt, Th_final, Tc_final


if __name__ == "__main__":
    seed_everything(7777)
    target_temp_in_C = -5.00
    volt_ref, amp_ref = 0.0, 10.0
    nc = neural_controller(T_ref=target_temp_in_C,
                           hidden_units=5,
                           bias=False,
                           lrate=1e-3)

    def p_controller(t, Th_arr, Tc_arr):
        print((Tc_arr[-1] - target_temp_in_C) @ u_V)
        return min(12.00,
                   0.5 * (Tc_arr[len(Tc_arr) - 1] - target_temp_in_C)) @ u_V

    def volt_input(t, Th_arr, Tc_arr):
        return volt_ref @ u_V

    def amp_input(t, Th_arr, Tc_arr):
        return amp_ref @ u_A

    res = []
    dt = 0.1
    step = int(10.0 / dt)
    for i in range(step):
        v, th, tc = test_control_algo(volt_input, voltage_src=True)
        volt_ref += dt
        res.append([v, th, tc])
        print("V: %f, Th: %f, Tc: %f" % (v, th, tc))
    res = np.array(res)
    np.save("system_identification", res)
    # test_control_algo(lambda t, T : 2.1@u_A, voltage_src = False)
    plt.show()
