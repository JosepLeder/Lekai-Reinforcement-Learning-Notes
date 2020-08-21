import numpy as np
import torch
import gym
import argparse
import os

from matplotlib import pyplot as plt
from parallel_env import MultiEnv, ParaEnv
from agent import A2C, DDPG, REINFORCE
from visualize import write_result

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default="REINFORCE")  # Policy name (TD3, DDPG or OurDDPG)
    parser.add_argument("--env", default="Humanoid-v2")  # OpenAI gym environment name
    parser.add_argument("--seed", default=0, type=int)  # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--start_timesteps", default=25e3, type=int)  # Time steps initial random policy is used
    parser.add_argument("--eval_freq", default=5e3, type=int)  # How often (time steps) we evaluate
    parser.add_argument("--max_timesteps", default=1e6, type=int)  # Max time steps to run environment
    parser.add_argument("--expl_noise", default=0.1)  # Std of Gaussian exploration noise
    parser.add_argument("--batch_size", default=256, type=int)  # Batch size for both actor and critic
    parser.add_argument("--n_steps", default=5, type=int)  # Number of steps per update
    parser.add_argument("--n_processes", default=16, type=int)  # Number of parallel environments
    parser.add_argument("--discount", default=0.99)  # Discount factor
    parser.add_argument("--tau", default=0.005)  # Target network update rate
    parser.add_argument("--policy_noise", default=0.2)  # Noise added to target policy during critic update
    parser.add_argument("--noise_clip", default=0.5)  # Range to clip target policy noise
    parser.add_argument("--policy_freq", default=2, type=int)  # Frequency of delayed policy updates
    parser.add_argument("--save_model", action="store_true")  # Save model and optimizer parameters
    parser.add_argument("--load_model", default="")  # Model load file name, "" doesn't load, "default" uses file_name
    args = parser.parse_args()

    file_name = f"{args.policy}_{args.env}_{args.seed}"
    print("---------------------------------------")
    print(f"Policy: {args.policy}, Env: {args.env}, Seed: {args.seed}")
    print("---------------------------------------")

    if not os.path.exists("./results"):
        os.makedirs("./results")

    if args.save_model and not os.path.exists("./models"):
        os.makedirs("./models")

    env = gym.make(args.env)

    # Set seeds
    env.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_episode_timesteps = env._max_episode_steps
    max_action = float(env.action_space.high[0])

    kwargs = {
        "state_dim": state_dim,
        "action_dim": action_dim,
        "max_action": max_action,
        "discount": args.discount,
        "tau": args.tau,
    }

    # Initialize policy
    # if args.policy == "TD3":
    #     # Target policy smoothing is scaled wrt the action scale
    #     kwargs["policy_noise"] = args.policy_noise * max_action
    #     kwargs["noise_clip"] = args.noise_clip * max_action
    #     kwargs["policy_freq"] = args.policy_freq
    #     policy = TD3.TD3(**kwargs)
    if args.policy == "A2C":
        envs = ParaEnv(args.env, args.n_processes, args.seed)
        policy = A2C.A2C(env.observation_space, env.action_space, args.discount, args.tau, max_episode_timesteps)
        x, y = policy.run(envs, file_name, args)
        write_result(args.env + "_A2C.json", x, y)

    elif args.policy == "DDPG":
        policy = DDPG.DDPG(**kwargs)
        x, y = policy.run(env, file_name, args)
        write_result(args.env + "_DDPG.json", x, y)

    elif args.policy == "REINFORCE":
        args.n_steps = 5
        args.n_processes = 16
        envs = ParaEnv(args.env, args.n_processes, args.seed)
        policy = REINFORCE.REINFORCE(env.observation_space, env.action_space, args.discount, args.tau,
                                     args.n_steps, args.n_processes, max_episode_timesteps)
        x, y = policy.run(envs, file_name, args)
        write_result(args.env + "_REINFORCE.json", x, y)

    else:
        x, y = None, None

    print(x)
    print(y)
    plt.figure()
    plt.xlabel('Timesteps')
    plt.ylabel('Reward')
    plt.plot(x, y)
    plt.show()
