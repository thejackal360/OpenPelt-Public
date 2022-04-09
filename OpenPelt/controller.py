from abc import ABC, abstractmethod

import random
from PySpice.Unit import u_V
from OpenPelt import TECPlate

try:
    from .neural_networks import MLP
    from torch import tensor
    INCLUDE_TORCH = True
except ImportError:
    INCLUDE_TORCH = False
    print("Warning: cannot import torch")

# RL constants
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 50

# Note: Ensure these are sync'd with OpenPelt's constant definitions
DEFAULT_TEMP_SENSOR_SAMPLES_PER_SEC = 1.00
DEFAULT_SIMULATION_TIMESTEPS_PER_SENSOR_SAMPLE = 2.00


class controller(ABC):
    """
    Abstract controller class. Used for implementing things like PID
    controllers.
    """

    def set_seqr(self, seqr):
        """
        Set the sequencer in use. Intended to be an instance of a subclass of
        sequencer.

        @param seqr: sequencer object in use
        """

        self.seqr = seqr

    def controller_f(self, t, sensor_dict):
        """
        Function called on each timestep to get the controller output. Output
        is interpreted as either a current or voltage, but this is not the
        responsibility of the controller class to specify.

        @param t: t is the current timestep.

        @param sensor_dict: sensor_dict maps "th" or "tc" strings to
        an array of historical values of hot side and cold side temperatures.

        @return: Output value to drive TEC circuit model
        """

        self.ref = self.seqr.get_ref()
        return self._controller_f(t, self.ref, sensor_dict)

    @abstractmethod
    def _controller_f(self, t, ref, sensor_dict):
        """
        Abstract method called by controller_f. Reference value is explicitly
        specified here. This function is internal to the abstract controller
        class.

        @param t: t is the current timestep.

        @param ref: reference temperature to which the controller steers the respective
        plate's heat sink

        @param sensor_dict: sensor_dict maps "th" or "tc" strings to
        an array of historical values of hot side and cold side temperatures.
        """
        pass

if INCLUDE_TORCH:
    class fake_neural_controller(controller):

        def __init__(self, seqr):
            self.seqr = seqr

            self.net = MLP(input_units=3,
                           hidden_units=64,
                           output_units=1,
                           bias=True).eval()

        def scale(self, x, var=(-100, 100), feature_range=(-1, 1)):
            x_std = (x - var[0]) / (var[1] - var[0])
            x_scaled = (x_std * (feature_range[1] - feature_range[0]) +
                        feature_range[0])
            return x_scaled

        def _controller_f(self, t, ref, sensor_dict):
            th = self.scale(sensor_dict['th'][-1])
            tc = self.scale(sensor_dict['tc'][-1])
            ref = self.scale(ref)
            self.state = tensor([th, tc, ref])
            v = self.net(self.state).detach().numpy()[0]
            return v @ u_V


class random_agent_controller(controller):

    def __init__(self, seqr):
        self.seqr = seqr
        self.ref = 0.0
        self.volt = [-6.0 + i for i in range(23)]
        self.total_r = 0
        self.freeze = False

    def take_action(self):
        return random.randint(0, 22)

    def reward(self):
        if self.state[0] == self.ref:
            self.freeze = True
            return 1
        else:
            return -1

    def agent(self, state):
        action = self.volt[self.take_action()]
        self.r = self.reward()
        self.total_r += self.r
        return action

    def _controller_f(self, t, ref, sensor_dict):
        th = sensor_dict['th'][-1]
        tc = sensor_dict['tc'][-1]
        self.ref = ref
        self.state = [th, tc, ref]
        v = self.agent(self.state)
        if self.freeze:
            v = 6.0
        return v


class bang_bang_controller(controller):
    """
    Bang-bang controller implementation. Inherits from abstract controller
    class.
    """

    def __init__(self, seqr):
        """
        Initialize controller. Need to specify reference value sequencer
        instance.

        @param seqr: sequencer object in use
        """
        self.seqr = seqr

    def _controller_f(self, t, ref, sensor_dict):
        """
        Bang-bang controller function. Check most recent hot plate temperature
        and drive 14.00 (usually volts) if below ref.

        @param t: t is the current timestep.

        @param ref: reference temperature to which the controller steers the respective
        plate's heat sink

        @param sensor_dict: sensor_dict maps "th" or "tc" strings to
        an array of historical values of hot side and cold side temperatures.

        @return: Output value to drive TEC circuit model
        """
        if sensor_dict["th"][-1] < ref:
            return 14.00
        else:
            return 0.00


class pid_controller(controller):
    """
    PID controller implementation. Inherits from abstract controller class.
    """

    def __init__(self, seqr, kp, ki, kd, plate_select,
                 samples_per_sec = DEFAULT_TEMP_SENSOR_SAMPLES_PER_SEC,
                 simulation_timesteps = DEFAULT_SIMULATION_TIMESTEPS_PER_SENSOR_SAMPLE):
        """
        Initialize controller. Specify proportional gain kp, integral gain ki,
        differential gain kd, and selected plate (plate_select). Need to
        specify reference value sequencer (seqr) instance as well

        @param seqr: sequencer object in use

        @param kp: proportional gain

        @param ki: integral gain

        @param kd: differential gain

        @param plate_select: selected plate's corresponding heat sink whose
        temperature we are controlling (TECPlate.HOT_SIDE or TECPlate.COLD_SIDE)

        @param samples_per_sec: the number of samples taken by the sensor per
        second

        @param simulation_timesteps: the number of simulation timesteps per
        sensor sample
        """
        # https://en.wikipedia.org/wiki/PID_controller
        self.seqr = seqr
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.dt = (1.00/(samples_per_sec * simulation_timesteps))
        self.prev_err = 0
        self.integral = 0
        self.plate_select = plate_select

    def _controller_f(self, t, ref, sensor_dict):
        """
        Clamp outputs at 16.00 at the high end and -6.00 at the low end.

        @param t: t is the current timestep.

        @param ref: reference temperature to which the controller steers the respective
        plate's heat sink

        @param sensor_dict: sensor_dict maps "th" or "tc" strings to
        an array of historical values of hot side and cold side temperatures.

        @return: Output value to drive TEC circuit model
        """
        error = ref - sensor_dict["th" if self.plate_select
                                  == TECPlate.HOT_SIDE else "tc"][-1]
        proportional = error
        self.integral = self.integral + (error * self.dt)
        derivative = (error - self.prev_err) / self.dt
        output = (self.kp * proportional) + \
                 (self.ki * self.integral) + \
                 (self.kd * derivative)
        output = min(16.00, max(-6.00, output))
        self.prev_err = error
        return output
