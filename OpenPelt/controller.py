from abc import ABC, abstractmethod

import random
import math
import numpy as np

import torch
from torch import nn
from torch.optim import RMSprop, Adam

from .neural_nets import MLP, Transition, ReplayMemory
from PySpice.Unit import u_V

# RL constants
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 50


class controller(ABC):
    """
    Abstract controller class. Used for implementing things like PID
    controllers.
    """

    """
    Set the sequencer in use. Intended to be an instance of a subclass of
    sequencer.
    """
    def set_seqr(self, seqr):
        self.seqr = seqr

    def controller_f(self, t, sensor_dict):
        """
        Function called on each timestep to get the controller output. Output
        is interpreted as either a current or voltage, but this is not the
        responsibility of the controller class to specify.

        t is the current timestep. sensor_dict maps "th" or "tc" strings to
        an array of historical values of hot side and cold side temperatures.
        """

        self.ref = self.seqr.get_ref()
        return self._controller_f(t, self.ref, sensor_dict)

    @abstractmethod
    def _controller_f(self, t, ref, sensor_dict):
        """
        Abstract method called by controller_f. Reference value is explicitly
        specified here. This function is internal to the abstract controller
        class.
        """
        pass


# TODO
class dqn_controller(controller):

    def __init__(self,
                 seqr,
                 n_actions=23,
                 gamma=0.999):
        self.seqr = seqr
        self.n_actions = n_actions
        self.gamma = gamma
        self.adam = False
        self.volt = [-6.0 + i for i in range(n_actions)]
        # self.volt = [0.0 + i for i in range(n_actions)]

        self.policy_net = MLP(input_units=3,
                              hidden_units=32,
                              output_units=self.n_actions,
                              bias=True)
        self.target_net = MLP(input_units=3,
                              hidden_units=32,
                              output_units=self.n_actions,
                              bias=True)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        if self.adam:
            self.optimizer = Adam(self.policy_net.parameters(),
                                  lr=1e-3,
                                  weight_decay=1e-5)
        else:
            self.optimizer = RMSprop(self.policy_net.parameters(),
                                     lr=5e-4)
        # self.criterion = nn.SmoothL1Loss()
        self.criterion = nn.MSELoss()
        self.memory = ReplayMemory(100000)
        self.steps_done = 0
        self.batch_size = 64
        self.iter = 1
        self.opt_flag = False
        self.done = False
        self.device = torch.device("cpu")
        self.R = 0
        self.num_episode = 0
        self.delta_old = None
        # self.writer = SummaryWriter()
        self.v_hist = []
        self.th_hist = []
        self.tc_hist = []

    def reset(self):
        self.iter = 1
        self.done = False
        self.opt_flag = False
        self.delta_old = None
        self.R = 0
        self.v_hist = []
        self.th_hist = []
        self.tc_hist = []

    def select_action(self, state):
        sample = random.random()
        eps_threshold = (EPS_END + (EPS_START - EPS_END) *
                         math.exp(-1. * self.steps_done / EPS_DECAY))
        self.steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                return self.policy_net(state).max(1)[1].view(1, 1)
        else:
            return torch.tensor([[random.randrange(self.n_actions)]],
                                dtype=torch.long)

    def optimize_model(self):
        if len(self.memory) < self.batch_size:
            return
        transitions = self.memory.sample(self.batch_size)

        batch = Transition(*zip(*transitions))

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                batch.next_state)),
                                      device=self.device,
                                      dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                           if s is not None])

        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        state_action_values = self.policy_net(state_batch).gather(1,
                                                                  action_batch)
        next_state_values = torch.zeros(self.batch_size, device=self.device)
        next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()

        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        loss = self.criterion(state_action_values,
                              expected_state_action_values.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            if param.requires_grad:
                param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
        # self.writer.add_scalar('loss'+str(self.num_episode),
        #                        loss.item(),
        #                        self.iter)

    def sigmoid(self, x):
        return 1. / (1. + np.exp(-x))

    def reward_(self, x, x_ref):
        # delta_new = self.sigmoid(x_ref[0, 0] - x[0, 0])
        # if self.delta_old is None:
        #     self.delta_old = delta_new
        #     return torch.tensor([0.0])

        # if self.volt[action] < -1:
        #     return torch.tensor([-0.2])
        # D = self.sigmoid(delta_new - self.delta_old)
        # if D > 0.5:
        #     return torch.tensor([-0.05])
        # elif D <= 0.5:
        #     return torch.tensor([0.1])
        # elif delta_new <= 0.01:
        #     return torch.tensor([0.2])
        # else:
        #     return torch.tensor([0.0])
        return torch.tensor([(x[0, 0] - x_ref[0, 0])**2])

    def reward1(self, x, x_ref):
        if x[0, 1] > 40:
            return torch.tensor([-.5])
        elif x[0, 0] >= 0 and x[0, 0] <= 30:
            return torch.tensor([.1])
        elif x[0, 0] > 30 and x[0, 0] < 40:
            return torch.tensor([.5])
        elif x[0, 0] >= 39.5 and x[0, 0] <= 40.5:
            return torch.tensor([5])
        elif x[0, 0] > 43. and x[0, 1] > 24:
            return torch.tensor([-.9])
        else:
            return torch.tensor([-.9])

    def terminate(self, ref):
        if self.state[0, 0] > (ref.item() + 1.)\
                or self.state[0, 1] > ref.item()\
                or self.iter == 4000:
            self.r = torch.tensor([-10])
            self.done = True

    def dql_train(self, x, ref):
        ref = torch.tensor(ref).view(-1, 1)

        if self.opt_flag is False and self.done is False:
            # print("ACTION", self.done)
            self.action = self.select_action(self.state)
            self.r = self.reward_(self.state, ref)
            self.R += self.r.item() * self.gamma**self.iter

            self.terminate(ref)

            self.iter += 1
            self.opt_flag = True

        elif self.opt_flag is True:
            # print("OPTIMIZE", self.done)
            if not self.done:
                self.next_state = x
            else:
                self.next_state = None

            self.memory.push(self.state,
                             self.action,
                             self.next_state,
                             self.r)

            self.state = self.next_state

            self.optimize_model()

            self.iter += 1
            self.opt_flag = False

        else:
            self.iter += 1
        return self.volt[self.action.item()]

    def _controller_f(self, t, ref, sensor_dict):
        th = sensor_dict['th'][-1]
        tc = sensor_dict['tc'][-1]
        if self.iter == 1:
            self.state = torch.tensor([0.0, 0.0, 0.0]).view(1, 3)
        else:
            self.state = torch.tensor([th, tc, ref]).view(1, 3)
        v = self.dql_train(self.state, ref)
        # print("Th: %f, Tc: %f, V: %f, REF: %f, R: %f, DONE: %d" % (th, tc, v,
        #                                                            ref,
        #                                                            self.r,
        #                                                            self.done))
        if self.done is False:
            self.v_hist.append(min(16.0, max(-6.0, v)))
            self.th_hist.append(sensor_dict['th'][-1])
            self.tc_hist.append(sensor_dict['tc'][-1])

            # self.writer.add_scalar('t-hot'+str(self.num_episode),
            #                        th,
            #                        self.iter)
            # self.writer.add_scalar('volt'+str(self.num_episode),
            #                        v,
            #                        self.iter)
            # self.writer.add_scalar('reward'+str(self.num_episode),
            #                        self.R,
            #                        self.iter)

        return min(16.0, max(-6.0, v)) @ u_V


class random_controller(controller):

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
        self.state = torch.tensor([th, tc, ref])
        v = self.net(self.state).detach().numpy()[0]
        print("Th: %f, Tc: %f, REF: %f, V: %f" % (th, tc, ref, v))
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
        """
        self.seqr = seqr

    def _controller_f(self, t, ref, sensor_dict):
        """
        Bang-bang controller function. Check most recent hot plate temperature
        and drive 14.00 (usually volts) if below ref.
        """
        if sensor_dict["th"][-1] < ref:
            return 14.00
        else:
            return 0.00


class pid_controller(controller):
    """
    PID controller implementation. Inherits from abstract controller class.
    """

    def __init__(self, seqr, kp, ki, kd, plate_select):
        """
        Initialize controller. Specify proportional gain kp, integral gain ki,
        differential gain kd, and selected plate (plate_select). Need to
        specify reference value sequencer instance as well
        """
        # https://en.wikipedia.org/wiki/PID_controller
        self.seqr = seqr
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.dt = (1.00/(TEMP_SENSOR_SAMPLES_PER_SEC
                   * SIMULATION_TIMESTEPS_PER_SENSOR_SAMPLE))
        self.prev_err = 0
        self.integral = 0
        self.plate_select = plate_select

    def _controller_f(self, t, ref, sensor_dict):
        """
        Clamp outputs at 16.00 at the high end and -6.00 at the low end.
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
        print("V: %f" % (output))
        self.prev_err = error
        return output
