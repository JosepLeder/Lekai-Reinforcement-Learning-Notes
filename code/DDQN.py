import gym
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T


# set up environment
env = gym.make('CartPole-v1')

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display
plt.ion()

# set up device
device = torch.device("cpu")

# define transition
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


# Create a queue to store transistions and use it for experience replay
class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = deque(maxlen=capacity)
        self.position = 0

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DDQN(nn.Module):

    # use simple MLP network as Q-Function
    def __init__(self, inputs, outputs):
        super(DDQN, self).__init__()
        self.l1 = nn.Linear(inputs, inputs * 6)
        self.l2 = nn.Linear(inputs * 6, inputs * 12)
        self.head = nn.Linear(inputs * 12, outputs)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        return self.head(x.view(x.size(0), -1))


def state2tensor(state):
    return torch.tensor(state, dtype=torch.float).unsqueeze(0).to(device)


# set up hyperparameters
MEM_REPLAY_SIZE = 150000
BATCH_SIZE = 128
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.08
EPS_DECAY = 200
EVALUATE_FREQUENCY = 20
ALTER_TARGET_UPDATE_RATE = 0.995


# Get number of actions from gym action and state space
random_state = env.reset()
n_states = random_state.size
n_actions = env.action_space.n


# Create network to represent Q-Function
current_net = DDQN(n_states, n_actions).to(device)
target_net = DDQN(n_states, n_actions).to(device)
target_net.load_state_dict(current_net.state_dict())
target_net.eval()

optimizer = optim.Adam(current_net.parameters())
memory = ReplayMemory(MEM_REPLAY_SIZE)


steps_cnt = 0


# Using epsilon greedy policy to select an action
def select_action(state):
    global steps_cnt
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_cnt / EPS_DECAY)
    steps_cnt += 1
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return current_net(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)


# Select the action with most rewards
def select_best_action(state):
    with torch.no_grad():
        return current_net(state).max(1)[1].view(1, 1)


episode_durations = []
evaluate_performence = []


def plot_durations():
    plt.figure(2)
    plt.clf()
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy(), label="Explore Score")
    plt.plot([x * EVALUATE_FREQUENCY for x in range(0, len(evaluate_performence))], 
            evaluate_performence, label="Optimal Socre")
    # Compute and plot current 10 episodes averages
    if len(durations_t) >= 10:
        means = durations_t.unfold(0, 10, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(9), means))
        plt.plot(means.numpy(), label="Average Explore Score")
    plt.legend()
    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        display.clear_output(wait=True)
        display.display(plt.gcf())
    


def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to current_net
    state_action_values = current_net(state_batch).gather(1, action_batch)

    # Compute argmax(a{t+1})[Q(s_{t+1}, a_{t+1})] for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute MSE loss
    loss = F.mse_loss(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()

    # clip the graient to avoid gradient gradient explosion
    for param in current_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()


def evaluate_model():
    durations = []
    for i_episode in range(100):
        # Initialize the environment and state
        state = env.reset()
        state = state2tensor(state)
        for t in count():
            # Select the action with the most rewards
            action = select_best_action(state)
            next_state, reward, done, _ = env.step(action.item())
            if done: next_state = None
            else: next_state = state2tensor(next_state)
            state = next_state
            
            if done or t + 1 >= 2000:
                durations.append(t + 1)
                break
    mean = np.mean(durations)
    evaluate_performence.append(mean)
    if mean > 195:
        print("Solved! Mean scores: {}".format(mean))
        return True
    else:
        print("Unsolved! Mean scores: {}".format(mean))
        return False


num_max_episodes = 1000
keys = current_net.state_dict().keys()

for i_episode in range(num_max_episodes):
    # Initialize the environment and state   
    state = env.reset()
    state = state2tensor(state)
    for t in count():
        # Select and perform an action
        action = select_action(state)
        next_state, reward, done, _ = env.step(action.item())
        reward = torch.tensor([reward], device=device)
        if done: next_state = None
        else: next_state = state2tensor(next_state)
        # Store the transition in memory
        memory.push(state, action, next_state, reward)

        # Move to the next state
        state = next_state

        # Perform one step of the optimization (on the target network)
        optimize_model()

        # Update the target network using alternative target network method
        # phi = tau * phi + (1 - tau) * phi_updated
        target_state = target_net.state_dict()
        policy_state = current_net.state_dict()
        for key in keys:
            target_state[key] = ALTER_TARGET_UPDATE_RATE * target_state[key] + (1 - ALTER_TARGET_UPDATE_RATE) * policy_state[key]
        target_net.load_state_dict(target_state)

        if done:
            episode_durations.append(t + 1)
            plot_durations()
            break
    if i_episode % EVALUATE_FREQUENCY == 0 and evaluate_model():
        print("Train finished after {} episodes!".format(i_episode + 1))
        break

plot_durations()
env.render()
env.close()
plt.ioff()
plt.show()

