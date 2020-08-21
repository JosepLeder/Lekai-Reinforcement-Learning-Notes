import os
import json
import gym
import torch
from matplotlib import pyplot as plt
from gym.wrappers.monitoring.video_recorder import VideoRecorder


policy_list = ["A2C", "DDPG", "REINFORCE"]
env_list = ["HalfCheetah-v2"]


def write_result(filename, x, y):
    data = {"x": x, "y": y}
    with open("results/" + filename, 'w') as f:
        json.dump(data, f)


def load_result(filename):
    with open("results/" + filename, 'r') as f:
        data = json.load(f)
    return data['x'], data['y']


def save_video(agent, env_name, video_path):
    num_episodes = 0
    # set up environment
    env = gym.make(env_name)
    state = env.reset()
    state = torch.tensor(state)

    # set up video recoder
    video_recorder = VideoRecorder(env, video_path, enabled=video_path is not None)

    # load model


def plot_results():
    plt.figure()
    for policy in policy_list:
        for env in env_list:
            json_name = env + "_" + policy + ".json"
            x, y = load_result(json_name)
            plt.plot(x, y)
    plt.xlabel("Timesteps")
    plt.ylabel("Reward")
    plt.legend(policy_list)
    plt.show()


if __name__ == '__main__':
    plot_results()
