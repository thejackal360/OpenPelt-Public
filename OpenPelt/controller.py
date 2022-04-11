from abc import ABC, abstractmethod

import random
from .neural_networks import MLP
from PySpice.Unit import u_V
from torch import tensor
from OpenPelt import TECPlate

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

        @param ref: reference temperature to which the controller steers the
        respective plate's heat sink

        @param sensor_dict: sensor_dict maps "th" or "tc" strings to
        an array of historical values of hot side and cold side temperatures.
        """
        pass


class fake_neural_controller(controller):
    """
    This class implements a neural network controller. It inherits from the
    controller class and implements a _controller_f method that calls a Pytorch
    object (in this case a simple MLP network).
    """

    def __init__(self, seqr):
        """
        Set the sequencer in use. Intended to be an instance of a subclass of
        sequencer. Furthermore, it initializes the neural controller (neural
        network).

        @param seqr: sequencer object in use
        """
        self.seqr = seqr

        # Initialize an MLP (see file neural_networks.py for more details on
        # the MLP)
        self.net = MLP(input_units=3,
                       hidden_units=64,
                       output_units=1,
                       bias=True).eval()

    def scale(self, x, var=(-100, 100), feature_range=(-1, 1)):
        """
        It scales argument x in the interval defines by feature_range.

        @param x: input scalar to be normalized

        @param var: A tuple that contains the maximum and the minimum values
        that x can take

        @param feature_range: A tuple that containst the desired range of the
        scaled data

        @return: The normalized value of x in the interval feature_ranage

        """
        x_std = (x - var[0]) / (var[1] - var[0])
        x_scaled = (x_std * (feature_range[1] - feature_range[0]) +
                    feature_range[0])
        return x_scaled

    def _controller_f(self, t, ref, sensor_dict):
        """
        Clamp outputs at 16.00 at the high end and -6.00 at the low end.

        @param t: t is the current timestep.

        @param ref: reference temperature to which the controller steers the
        respective plate's heat sink

        @param sensor_dict: sensor_dict maps "th" or "tc" strings to
        an array of historical values of hot side and cold side temperatures.

        @return: Output value to drive TEC circuit model
        """
        th = self.scale(sensor_dict['th'][-1])
        tc = self.scale(sensor_dict['tc'][-1])
        ref = self.scale(ref)
        self.state = tensor([th, tc, ref])
        v = self.net(self.state).detach().numpy()[0]
        return v @ u_V


class random_agent_controller(controller):
    """
    This class implements a naive (random) agent. It demonstrates how the user
    can implement reinforcement learning algorithms using OpenPelt as a TEC
    simulator.
    """

    def __init__(self, seqr):
        """
        Set the sequencer in use. Intended to be an instance of a subclass of
        sequencer. Furthermore, it sets up all the total reward of the agent
        (total_r), the set of permited actions the agent can take (volt). The
        actions are voltage values that pass to the OpenPelt simulator and
        controll the temperature. The parameter freeze determines if the
        episode has ended meaenint the agent has succesfully set the
        temperature to the reference value.

        @param seqr: sequencer object in use
        """
        self.seqr = seqr
        self.ref = 0.0          # reference temperature value
        self.volt = [-6.0 + i for i in range(23)]   # voltages (actions)
        self.total_r = 0        # total reward
        self.freeze = False     # when the agent solves the problem freeze the actions

    def take_action(self):
        """
        This method randomly chooses an action based on a uniform distribution.

        @return The index (integer) of an action to be taken. This is the index
        of the volt variable that will determine the voltages the agent will
        pass to the OpenPelt model.

        """
        return random.randint(0, 22)

    def reward(self):
        """
        This method provides the instant reward to the agent based on its
        action.

        @param 1 in case the agent has set properly the temperature to the
        reference value, -1 otherwise.

        """
        if self.state[0] == self.ref:
            self.freeze = True
            return 1
        else:
            return -1

    def agent(self, state):
        """
        The agent method chooses an action, sets the voltage value based on
        that action, receives a reward and computes the total reward.

        @param state: The state is a 2x1 numpy array that contains the current
        temperature (sensor readout) and the reference temperature.

        @return action: This is a float scalar that represents the voltage
        value the agent passes to the OpenPelt model. Essentialy is the control
        signal.
        """
        action = self.volt[self.take_action()]
        self.r = self.reward()
        self.total_r += self.r
        return action

    def _controller_f(self, t, ref, sensor_dict):
        """
        Clamp outputs at 16.00 at the high end and -6.00 at the low end.

        @param t: t is the current timestep.

        @param ref: reference temperature to which the controller steers the
        respective plate's heat sink

        @param sensor_dict: sensor_dict maps "th" or "tc" strings to
        an array of historical values of hot side and cold side temperatures.

        @return: Output value to drive TEC circuit model
        """
        """
        Clamp outputs at 16.00 at the high end and -6.00 at the low end.

        @param t: t is the current timestep.

        @param ref: reference temperature to which the controller steers the
        respective plate's heat sink

        @param sensor_dict: sensor_dict maps "th" or "tc" strings to
        an array of historical values of hot side and cold side temperatures.

        @return: Output value to drive TEC circuit model
        """
        th = sensor_dict['th'][-1]
        tc = sensor_dict['tc'][-1]
        self.ref = ref
        self.state = [th, tc, ref]
        v = self.agent(self.state)
        # Once the agent sets correctly the temperature, we set the voltage
        # value to a constant (6.0) and wait until the simulation terminates.
        if self.freeze:
            v = 6.0
        return v @ u_V


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

        @param ref: reference temperature to which the controller steers the
        respective plate's heat sink

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
                 samples_per_sec=DEFAULT_TEMP_SENSOR_SAMPLES_PER_SEC,
                 simulation_timesteps=DEFAULT_SIMULATION_TIMESTEPS_PER_SENSOR_SAMPLE):
        """
        Initialize controller. Specify proportional gain kp, integral gain ki,
        differential gain kd, and selected plate (plate_select). Need to
        specify reference value sequencer (seqr) instance as well

        @param seqr: sequencer object in use

        @param kp: proportional gain

        @param ki: integral gain

        @param kd: differential gain

        @param plate_select: selected plate's corresponding heat sink whose
        temperature we are controlling (TECPlate.HOT_SIDE or
        TECPlate.COLD_SIDE)

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

        @param ref: reference temperature to which the controller steers the
        respective plate's heat sink

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
