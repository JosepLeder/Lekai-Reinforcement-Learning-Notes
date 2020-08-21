import numpy as np
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
import torch as th
from torch import nn
from typing import Generator, Optional, Union, Tuple, NamedTuple
from gym import spaces


def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module


def _flatten_helper(T, N, _tensor):
    return _tensor.view(T * N, *_tensor.size()[2:])


class TrajectoryBuffer(object):
    def __init__(self, num_steps, num_processes, obs_shape, action_space):
        self.obs = th.zeros(num_steps + 1, num_processes, *obs_shape)
        self.rewards = th.zeros(num_steps + 1, num_processes, 1)
        self.returns = th.zeros(num_steps + 1, num_processes, 1)
        self.action_log_probs = th.zeros(num_steps, num_processes, 1)
        self.actions = th.zeros(num_steps, num_processes, action_space.shape[0])
        self.masks = th.ones(num_steps + 1, num_processes, 1)

        # Masks that indicate whether it's a true terminal state
        # or time limit end state
        self.bad_masks = th.ones(num_steps + 1, num_processes, 1)

        self.num_steps = num_steps
        self.step = 0

    def to(self, device):
        self.obs = self.obs.to(device)
        self.rewards = self.rewards.to(device)
        self.returns = self.returns.to(device)
        self.action_log_probs = self.action_log_probs.to(device)
        self.actions = self.actions.to(device)
        self.masks = self.masks.to(device)
        self.bad_masks = self.bad_masks.to(device)

    def insert(self, obs, actions, action_log_probs, rewards, masks, bad_masks):
        self.obs[self.step + 1].copy_(obs)
        self.actions[self.step].copy_(actions)
        self.action_log_probs[self.step].copy_(action_log_probs)
        self.rewards[self.step + 1].copy_(rewards)
        self.masks[self.step + 1].copy_(masks)
        self.bad_masks[self.step + 1].copy_(bad_masks)

        self.step = (self.step + 1) % self.num_steps

    def after_update(self):
        self.obs[0].copy_(self.obs[-1])
        self.rewards[0].copy_(self.rewards[-1])
        self.masks[0].copy_(self.masks[-1])
        self.bad_masks[0].copy_(self.bad_masks[-1])

    def compute_returns(self, gamma):
        self.returns[-1] = self.rewards[0]
        for step in reversed(range(self.num_steps)):
            self.returns[step] = (self.returns[step + 1] * gamma * self.masks[step + 1] +
                                  self.rewards[step + 1]) * self.bad_masks[step + 1]


class ReplayBuffer(object):
    def __init__(self, state_dim, action_dim, max_size=int(1e6)):
        self.max_size = max_size
        self.pos = 0
        self.size = 0

        self.state = np.zeros((max_size, state_dim))
        self.action = np.zeros((max_size, action_dim))
        self.next_state = np.zeros((max_size, state_dim))
        self.reward = np.zeros((max_size, 1))
        self.not_done = np.zeros((max_size, 1))

        self.device = th.device("cpu")

    def add(self, state, action, next_state, reward, done):
        self.state[self.pos] = state
        self.action[self.pos] = action
        self.next_state[self.pos] = next_state
        self.reward[self.pos] = reward
        self.not_done[self.pos] = 1. - done

        self.pos = (self.pos + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)

        return (
            th.FloatTensor(self.state[ind]).to(self.device),
            th.FloatTensor(self.action[ind]).to(self.device),
            th.FloatTensor(self.next_state[ind]).to(self.device),
            th.FloatTensor(self.reward[ind]).to(self.device),
            th.FloatTensor(self.not_done[ind]).to(self.device)
        )


class RolloutBuffer(object):
    def __init__(self, num_steps, num_processes, obs_shape, action_space):
        self.obs = th.zeros(num_steps + 1, num_processes, *obs_shape)
        self.rewards = th.zeros(num_steps, num_processes, 1)
        self.value_preds = th.zeros(num_steps + 1, num_processes, 1)
        self.returns = th.zeros(num_steps + 1, num_processes, 1)
        self.action_log_probs = th.zeros(num_steps, num_processes, 1)
        self.actions = th.zeros(num_steps, num_processes, action_space.shape[0])
        self.masks = th.ones(num_steps + 1, num_processes, 1)

        # Masks that indicate whether it's a true terminal state
        # or time limit end state
        self.bad_masks = th.ones(num_steps + 1, num_processes, 1)

        self.num_steps = num_steps
        self.step = 0

    def to(self, device):
        self.obs = self.obs.to(device)
        self.rewards = self.rewards.to(device)
        self.value_preds = self.value_preds.to(device)
        self.returns = self.returns.to(device)
        self.action_log_probs = self.action_log_probs.to(device)
        self.actions = self.actions.to(device)
        self.masks = self.masks.to(device)
        self.bad_masks = self.bad_masks.to(device)

    def insert(self, obs, actions, action_log_probs,
               value_preds, rewards, masks, bad_masks):
        self.obs[self.step + 1].copy_(obs)
        self.actions[self.step].copy_(actions)
        self.action_log_probs[self.step].copy_(action_log_probs)
        self.value_preds[self.step].copy_(value_preds)
        self.rewards[self.step].copy_(rewards)
        self.masks[self.step + 1].copy_(masks)
        self.bad_masks[self.step + 1].copy_(bad_masks)

        self.step = (self.step + 1) % self.num_steps

    def after_update(self):
        self.obs[0].copy_(self.obs[-1])
        self.masks[0].copy_(self.masks[-1])
        self.bad_masks[0].copy_(self.bad_masks[-1])

    def compute_returns(self, next_value, gamma):
        self.returns[-1] = next_value
        for step in reversed(range(self.rewards.size(0))):
            self.returns[step] = (self.returns[step + 1] * gamma * self.masks[step + 1] + self.rewards[step]) * \
                                 self.bad_masks[step + 1] + (1 - self.bad_masks[step + 1]) * self.value_preds[step]


def sum_independent_dims(tensor: th.Tensor) -> th.Tensor:
    """
    Continuous actions are usually considered to be independent,
    so we can sum components of the ``log_prob`` or the entropy.
    :param tensor: (th.Tensor) shape: (n_batch, n_actions) or (n_batch,)
    :return: (th.Tensor) shape: (n_batch,)
    """
    if len(tensor.shape) > 1:
        tensor = tensor.sum(dim=1)
    else:
        tensor = tensor.sum()
    return tensor


