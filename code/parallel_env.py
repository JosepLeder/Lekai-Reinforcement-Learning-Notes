import gym
import torch
import numpy as np
from multiprocessing import Pool


class MultiEnv(object):
    def __init__(self, env_name, num_env):
        self.processes = []
        for i in range(num_env):
            env = gym.make(env_name)
            env.seed(0)
            self.processes.append(env)
        self.num_processes = num_env

    def reset(self):
        states = []
        for env in self.processes:
            states.append(env.reset())
        return torch.FloatTensor(states), [False] * self.num_processes

    def step(self, actions):
        next_state, reward, done, infos = [], [], [], []
        for env, action in zip(self.processes, actions):
            s, r, d, i = env.step(action)
            next_state.append(s)
            reward.append([r])
            done.append(d)
            infos.append(i)
        return torch.tensor(next_state), torch.tensor(reward), done, infos


class ParaEnv(MultiEnv):
    def __init__(self, env_name, num_env, seed):

        super().__init__(env_name, num_env)
        self.seed = seed
        for i in range(self.num_processes):
            self.processes[i].seed(seed + i)

    def seed(self, seed):
        [self.processes[i].seed(seed + i) for i in range(self.num_processes)]


if __name__ == '__main__':
    num_proc = 16
    parallel_envs = MultiEnv("HalfCheetah-v2", num_proc)
    action_space = gym.make("HalfCheetah-v2").action_space
    actions = [action_space.sample() for j in range(num_proc)]
    parallel_envs.reset()
    parallel_envs.step(actions)
    for i in range(1000):
        next_state, reward, done, infos = parallel_envs.step(actions)

    print(next_state)
    print(reward)
    print(done)
    print(infos)
