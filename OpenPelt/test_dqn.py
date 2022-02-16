import random
import math

import numpy as np
import matplotlib.pylab as plt

# from itertools import count
from initialize_randomness import seed_everything

from neural_nets import MLP, Transition, ReplayMemory

import torch
from torch import nn
from torch.optim import RMSprop, Adam


EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
GAMMA = 0.5

volt = [-6.0 + i for i in range(0, 23, 1)]
# volt = [-6.0, -2.0, -1.0, 0.0, 1.0, 2.0, 6.0, 12.0]

count = 0
batch_size = 10
steps_done = 0
n_actions = len(volt)
device = torch.device("cpu")

seed_everything(133)

criterion = nn.SmoothL1Loss()
# criterion = nn.MSELoss()
memory = ReplayMemory(100000)

policy_net = MLP(input_units=3,
                 hidden_units=20,
                 output_units=n_actions,
                 bias=True)
target_net = MLP(input_units=3,
                 hidden_units=20,
                 output_units=n_actions,
                 bias=True)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

# optimizer = Adam(policy_net.parameters(), lr=1e-3)
optimizer = RMSprop(policy_net.parameters(), lr=5e-4)

base = "../results/"
systemh = np.load(base+"op_point_voltage_volt_th_characterization.npy")
systemc = np.load(base+"op_point_voltage_volt_tc_characterization.npy")
systemH = np.round(systemh, 1).astype('f')
systemC = np.round(systemc, 1).astype('f')

REF_VAL = 38
ref = torch.tensor([REF_VAL]).view(-1, 1)


def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = (EPS_END + (EPS_START - EPS_END) *
                     math.exp(-1. * steps_done / EPS_DECAY))
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[random.randrange(n_actions)]],
                            dtype=torch.long)


def optimize_model():
    if len(memory) < batch_size:
        return
    transitions = memory.sample(batch_size)

    batch = Transition(*zip(*transitions))

    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)),
                                  device=device,
                                  dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                       if s is not None])

    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    state_action_values = policy_net(state_batch).gather(1, action_batch)
    next_state_values = torch.zeros(batch_size, device=device)
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()

    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    loss = criterion(state_action_values,
                     expected_state_action_values.unsqueeze(1))

    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()


def reward_(x, x_ref):
    delta = np.abs(x_ref - x)
    if delta <= 0.5:
        return torch.tensor([1000.0])
    elif delta >= 1 and delta <= 2:
        return torch.tensor([-100.0])
    else:
        return torch.tensor([-500.0])
    # return -torch.tensor([float((x_ref - x)**2)])


def env_step(action):
    global count
    act = min(16.0, max(-6.0, volt[action.item()]))
    # act = volt[action.item()]
    done = False

    idx = np.where(systemH[0, :] == act)[0]
    temp_h = float(systemH[1, idx[0]])
    reward = reward_(temp_h, REF_VAL)

    idx = np.where(systemC[0, :] == act)[0]
    temp_c = float(systemC[1, idx[0]])

    if temp_h > REF_VAL + 2 or temp_h < REF_VAL - 2:
        count += 1

    if count > 5:
        done = True

    if temp_c > 30:
        done = True
    new_state = torch.tensor([temp_h, temp_c, REF_VAL]).view(1, -1)
    return new_state, reward, done


def plot_(ax, th, tc):
    ax.plot(th, 'r')
    ax.plot(tc, 'b')
    plt.axhline(REF_VAL, c='k')
    ax.set_ylim([-25, 100])


def controller(episodes=100):
    global count

    th_hist = []
    tc_hist = []
    cum_rew = []
    # fig = plt.figure(figsize=(9, 3))
    # ax = fig.add_subplot(111)

    R = 0
    for e in range(episodes):
        th = []
        tc = []
        state = torch.tensor([0.0, 0.0, 0.0]).view(1, -1)
        for t in range(150):
            th.append(state[0, 0])
            tc.append(state[0, 1])
            action = select_action(state)
            new_state, r, done = env_step(action)
            R += r.item() * GAMMA**t

            if not done:
                next_state = new_state
            else:
                next_state = None

            memory.push(state, action, next_state, r)

            state = next_state

            optimize_model()
            if done:
                print("DONE")
                break
        print("Episode %d exhausted" % e)
        cum_rew.append(R)
        th_hist.append(th)
        tc_hist.append(tc)
        count = 0
        # plot_(ax, th, tc)
        if e % 10 == 0:
            target_net.load_state_dict(policy_net.state_dict())

    fig = plt.figure(figsize=(9, 3))
    ax = fig.add_subplot(111)
    for i in range(episodes):
        plot_(ax, th_hist[i], tc_hist[i])

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(cum_rew, 'k', lw=2)

    return th_hist[-1], tc_hist[-1]


if __name__ == '__main__':
    th, tc = controller(episodes=100)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(th, 'r')
    ax.plot(tc, 'b')
    ax.set_ylim([-25, 100])
    ax.set_xlim([0, 150])
    plt.axhline(REF_VAL, c='k')
    plt.show()
