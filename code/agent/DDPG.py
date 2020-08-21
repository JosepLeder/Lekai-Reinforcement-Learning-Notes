import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from copy import deepcopy
from utils import ReplayBuffer

# set up device
device = torch.device("cpu")


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()

        self.max_action = max_action

        self.l1 = nn.Linear(state_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, action_dim)

    def forward(self, state):
        x = F.relu(self.l1(state))
        x = F.relu(self.l2(x))
        return self.max_action * torch.tanh(self.l3(x))


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        self.l1 = nn.Linear(state_dim + action_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, 1)

    def forward(self, state, action):
        x = F.relu(self.l1(torch.cat([state, action], 1)))
        x = F.relu(self.l2(x))
        return self.l3(x)


class DDPG(object):
    def __init__(self, state_dim, action_dim, max_action, discount=0.99, tau=0.005):
        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters())

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters())

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action
        self.discount = discount
        self.tau = tau

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        return self.actor(state).cpu().data.numpy().flatten()

    def run_episode(self, replay_buffer, batch_size=100):
        # random sample from replay buffer
        state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)

        # compute target Q value
        target_Q = self.critic_target(next_state, self.actor_target(next_state))
        # split the target_Q from network, so that it becomes a constant which has no gradient
        target_Q = reward + (not_done * self.discount * target_Q).detach()

        # compute current Q value
        current_Q = self.critic(state, action)

        # compute critic loss
        critic_loss = F.mse_loss(current_Q, target_Q)

        # optimize the critic network
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # compute actor loss
        actor_loss = -self.critic(state, self.actor(state)).mean()

        # optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        # update the target network by alternative target network method
        # need to be optimized
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def save(self, filename):
        torch.save(self.critic.state_dict(), filename + "_critic")
        torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer")

        torch.save(self.actor.state_dict(), filename + "_actor")
        torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")

    def load(self, filename):
        self.critic.load_state_dict(torch.load(filename + "_critic"))
        self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer"))
        self.critic_target = deepcopy(self.critic)

        self.actor.load_state_dict(torch.load(filename + "_actor"))
        self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))
        self.actor_target = deepcopy(self.actor)

    # Runs policy for X episodes and returns average reward
    # A fixed seed is used for the eval environment
    def eval_policy(self, env_name, seed, eval_episodes=10):
        eval_env = gym.make(env_name)
        eval_env.seed(seed + 100)

        avg_reward = 0.
        for _ in range(eval_episodes):
            state, done = eval_env.reset(), False
            while not done:
                action = self.select_action(np.array(state))
                state, reward, done, _ = eval_env.step(action)
                avg_reward += reward

        avg_reward /= eval_episodes

        print("---------------------------------------")
        print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
        print("---------------------------------------")
        return avg_reward

    def run(self, env, file_name, args):
        if args.load_model != "":
            policy_file = file_name if args.load_model == "default" else args.load_model
            self.load(f"./models/{policy_file}")

        replay_buffer = ReplayBuffer(self.state_dim, self.action_dim)

        # Evaluate untrained policy
        evaluations = [self.eval_policy(args.env, args.seed)]

        state, done = env.reset(), False
        episode_reward = 0
        episode_timesteps = 0
        episode_num = 0

        # Initialize data list for plot
        x_timesteps = []
        y_rewards = []
        for t in range(int(args.max_timesteps)):

            episode_timesteps += 1

            # Select action randomly or according to policy
            if t < args.start_timesteps:
                action = env.action_space.sample()
            else:
                action = (
                        self.select_action(np.array(state))
                        + np.random.normal(0, self.max_action * args.expl_noise, size=self.action_dim)
                ).clip(-self.max_action, self.max_action)

            # Perform action
            next_state, reward, done, _ = env.step(action)
            done_bool = float(done) if episode_timesteps < env._max_episode_steps else 0

            # Store data in replay buffer
            replay_buffer.add(state, action, next_state, reward, done_bool)

            state = next_state
            episode_reward += reward

            # Train agent after collecting sufficient data
            if t >= args.start_timesteps:
                self.run_episode(replay_buffer, args.batch_size)

            if done:
                # +1 to account for 0 indexing. +0 on ep_timesteps since it will increment +1 even if done=True
                print(
                    f"Total T: {t + 1}/{args.max_timesteps} Episode Num: {episode_num + 1} Reward: {episode_reward:.3f}")
                # Compute and reset plot info
                y_rewards.append(episode_reward)
                x_timesteps.append(t+1)
                # Reset environment
                state, done = env.reset(), False
                episode_reward = 0
                episode_timesteps = 0
                episode_num += 1

            # Evaluate episode
            if (t + 1) % args.eval_freq == 0:
                evaluations.append(self.eval_policy(args.env, args.seed))
                np.save(f"./results/{file_name}", evaluations)
                if args.save_model: self.save(f"./models/{file_name}")

        return x_timesteps, y_rewards


