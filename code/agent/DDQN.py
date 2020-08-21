import gym
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
from gym.wrappers.monitoring.video_recorder import VideoRecorder

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

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

steps_cnt = 0

episode_durations = []
evaluate_performance = []


# Create a queue to store tranitions and use it for experience replay
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


class DQN(nn.Module):

    # use simple MLP network as Q-Function
    def __init__(self, inputs, outputs):
        super(DQN, self).__init__()
        self.l1 = nn.Linear(inputs, inputs * 24)
        self.l2 = nn.Linear(inputs * 24, inputs * 24)
        self.head = nn.Linear(inputs * 24, outputs)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        return self.head(x.view(x.size(0), -1))


class Agent(object):

    def __init__(self, hp, env_name='CartPole-v1'):
        # set up environment
        self.env = gym.make(env_name)
        self.env_name = env_name

        # Get number of actions from gym action and state space
        random_state = self.env.reset()
        self.n_states = random_state.size
        self.n_actions = self.env.action_space.n

        # Create network to represent Q-Function
        self.current_net = DQN(self.n_states, self.n_actions).to(device)
        self.target_net = DQN(self.n_states, self.n_actions).to(device)
        self.target_net.load_state_dict(self.current_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.current_net.parameters())
        self.memory = ReplayMemory(hp.MEM_REPLAY_SIZE)

        self.steps_cnt = 0


class HyperParameters(object):

    def __init__(self, params):
        self.MEM_REPLAY_SIZE = params['MEM_REPLAY_SIZE']
        self.BATCH_SIZE = params['BATCH_SIZE']
        self.GAMMA = params['GAMMA']
        self.EPS_START = params['EPS_START']
        self.EPS_END = params['EPS_END']
        self.EPS_DECAY = params['EPS_DECAY']
        self.EVALUATE_FREQUENCY = params['EVALUATE_FREQUENCY']
        self.ALTER_TARGET_UPDATE_RATE = params['ALTER_TARGET_UPDATE_RATE']
        self.MAX_EPISODES = params['MAX_EPISODES']


def state2tensor(state):
    return torch.tensor(state, dtype=torch.float).unsqueeze(0).to(device)


# Using epsilon greedy policy to select an action
def select_action(state, agent, hp):
    global steps_cnt
    sample = random.random()
    eps_threshold = hp.EPS_END + (hp.EPS_START - hp.EPS_END) * \
                    math.exp(-1. * steps_cnt / hp.EPS_DECAY)
    steps_cnt += 1
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return agent.current_net(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[random.randrange(agent.n_actions)]], device=device, dtype=torch.long)


# Select the action with most rewards
def select_best_action(state, agent):
    with torch.no_grad():
        return agent.current_net(state).max(1)[1].view(1, 1)


def plot_durations(scores_to_win, hp):
    plt.figure(2)
    plt.clf()
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    scores_to_win = torch.tensor([scores_to_win] * len(episode_durations))
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy(), label="Explore Reward")
    plt.plot([x * hp.EVALUATE_FREQUENCY for x in range(0, len(evaluate_performance))],
             evaluate_performance, label="Optimal Score")
    plt.plot(scores_to_win.numpy(), label="Target Score")
    # Compute and plot current 10 episodes averages
    if len(durations_t) >= 10:
        means = durations_t.unfold(0, 10, 1).mean(1).view(-1)
        means = torch.cat((durations_t[0:9], means))
        plt.plot(means.numpy(), label="Average Explore Reward")
    plt.legend()
    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        display.clear_output(wait=True)
        display.display(plt.gcf())


def optimize_model(agent, hp):
    if len(agent.memory) < hp.BATCH_SIZE:
        return
    transitions = agent.memory.sample(hp.BATCH_SIZE)
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
    state_action_values = agent.current_net(state_batch).gather(1, action_batch)

    # Compute argmax(a{t+1})[Q(s_{t+1}, a_{t+1})] for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    next_state_values = torch.zeros(hp.BATCH_SIZE, device=device)
    next_state_values[non_final_mask] = agent.target_net(non_final_next_states).max(1)[0].detach()
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * hp.GAMMA) + reward_batch
    # Compute MSE loss
    loss = F.mse_loss(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    agent.optimizer.zero_grad()
    loss.backward()

    # clip the gradient to avoid gradient gradient explosion
    for param in agent.current_net.parameters():
        param.grad.data.clamp_(-1, 1)
    agent.optimizer.step()


def evaluate_model(agent, scores_to_win):
    durations = []
    for i_episode in range(1):
        # Initialize the environment and state
        state = agent.env.reset()
        state = state2tensor(state)
        total_reward = 0
        for t in count():
            # Select the action with the most rewards
            action = select_best_action(state, agent=agent)
            next_state, reward, done, _ = agent.env.step(action.item())
            total_reward += reward
            if done:
                next_state = None
            else:
                next_state = state2tensor(next_state)
            state = next_state

            if done or t + 1 >= 2000:
                durations.append(total_reward)
                break
    mean = np.mean(durations)
    evaluate_performance.append(mean)
    if mean > scores_to_win:
        print("Solved! Mean scores: {}".format(mean))
        return True
    else:
        print("Unsolved! Mean scores: {}".format(mean))
        return False


def get_scores_to_win(agent):
    try:
        scores_to_win = agent.env.unwrapped.reward_threshold
    except AttributeError:
        try:
            scores_to_win = agent.env.spec.reward_threshold
        except AttributeError:
            scores_to_win = agent.env.unwrapped.spec.reward_threshold
    return scores_to_win if scores_to_win is not None else -10


def get_reward(agent, state, next_state, reward, done, total_reward):
    if agent.env_name in ['CartPole-v0', 'CartPole-v1', 'Acrobot-v1']:
        return reward
    elif agent.env_name == 'MountainCar-v0':
        if done:
            return 210 + total_reward
        else:
            return abs(next_state[0] - state[0])[0]
    else:
        return 0


def save_video(agent, video_path):
    num_episodes = 0
    # video_recorder = None
    video_recorder = VideoRecorder(
        agent.env, video_path, enabled=video_path is not None)
    state = agent.env.reset()
    state = state2tensor(state)
    for t in count():
        agent.env.unwrapped.render()
        video_recorder.capture_frame()
        action = select_best_action(state=state, agent=agent)
        next_state, rew, done, info = agent.env.step(action.item())
        next_state = state2tensor(next_state)
        state = next_state

        if done:
            # save video of first episode
            print("Saved video.")
            video_recorder.close()
            video_recorder.enabled = False
            break


def train_model(params, env='CartPole-v1'):
    hp = HyperParameters(params)
    agent = Agent(hp, env_name=env)
    keys = agent.current_net.state_dict().keys()
    scores_to_win = get_scores_to_win(agent=agent)

    for i_episode in range(hp.MAX_EPISODES):
        # Initialize the environment and state
        state = agent.env.reset()
        state = state2tensor(state)
        total_reward = 0
        for t in count():
            # Select and perform an action
            action = select_action(state, agent, hp)
            next_state, reward, done, _ = agent.env.step(action.item())
            reward = get_reward(agent, state, next_state, reward, done, -t)
            total_reward += reward
            reward = torch.tensor([reward], device=device)
            if done:
                next_state = None
            else:
                next_state = state2tensor(next_state)
            # Store the transition in memory

            agent.memory.push(state, action, next_state, reward)

            # Move to the next state
            state = next_state

            # Perform one step of the optimization (on the target network)
            optimize_model(agent=agent, hp=hp)

            # Update the target network using alternative target network method
            # phi = tau * phi + (1 - tau) * phi_updated
            target_state = agent.target_net.state_dict()
            policy_state = agent.current_net.state_dict()
            for key in keys:
                target_state[key] = hp.ALTER_TARGET_UPDATE_RATE * target_state[key] + \
                                    (1 - hp.ALTER_TARGET_UPDATE_RATE) * policy_state[key]
            agent.target_net.load_state_dict(target_state)

            if done:
                episode_durations.append(total_reward)
                plot_durations(scores_to_win, hp=hp)
                break
        if i_episode % hp.EVALUATE_FREQUENCY == 0 and evaluate_model(agent=agent, scores_to_win=scores_to_win):
            print("Train finished after {} episodes!".format(i_episode + 1))
            break

    plot_durations(scores_to_win, hp=hp)
    save_video(agent=agent, video_path="video/" + "DDQN_" + env + ".mp4")
    agent.env.render()
    agent.env.close()
    plt.ioff()
    plt.show()


if __name__ == '__main__':
    train_model({'MEM_REPLAY_SIZE': 150000,
                 'BATCH_SIZE': 128,
                 'GAMMA': 0.999,
                 'EPS_START': 0.9,
                 'EPS_END': 0.08,
                 'EPS_DECAY': 200,
                 'EVALUATE_FREQUENCY': 20,
                 'ALTER_TARGET_UPDATE_RATE': 0.995,
                 'MAX_EPISODES': 1000})
