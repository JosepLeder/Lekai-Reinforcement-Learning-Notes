import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import TrajectoryBuffer, init
from distributions import DiagGaussianDistribution

# set up device
device = torch.device("cpu")


class Policy(nn.Module):
    def __init__(self, obs_shape, action_space):
        super(Policy, self).__init__()
        base = MLPBase
        self.base = base(obs_shape[0])

        num_outputs = action_space.shape[0]
        self.dist = DiagGaussianDistribution(self.base.output_size, num_outputs)

    def forward(self, inputs):
        raise NotImplementedError

    def select_actions(self, inputs, deterministic=False):

        actor_features = self.base(inputs)
        dist = self.dist(actor_features)

        if deterministic:
            action = dist.mode()
        else:
            action = dist.sample()

        action_log_probs = dist.log_probs(action)

        return action, action_log_probs

    def evaluate_actions(self, inputs, action):
        actor_features = self.base(inputs)
        dist = self.dist(actor_features)
        action_log_probs = dist.log_probs(action)

        return action_log_probs


class MLPBase(nn.Module):
    def __init__(self, num_inputs, hidden_size=64):
        super(MLPBase, self).__init__()
        self._hidden_size = hidden_size
        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), np.sqrt(2))

        self.actor = nn.Sequential(
            init_(nn.Linear(num_inputs, hidden_size)), nn.Tanh(),
            init_(nn.Linear(hidden_size, hidden_size)), nn.Tanh())
        self.train()

    def forward(self, inputs):
        policy_latent = self.actor(inputs)
        return policy_latent

    @property
    def output_size(self):
        return self._hidden_size


class REINFORCE(object):

    def __init__(self, observation_space, action_space, discount, tau, n_steps, n_processes, max_episode_timesteps):
        self.max_episode_timesteps = max_episode_timesteps
        self.observation_space = observation_space
        self.action_space = action_space
        self.n_steps = n_steps
        self.n_processes = n_processes
        self.state_dim = observation_space.shape[0]
        self.action_dim = action_space.shape[0]
        self.max_action = float(action_space.high[0])
        self.discount = discount
        self.learning_rate = 7e-4
        self.tau = tau
        self.max_grad_norm = 0.5

        self.policy = Policy(observation_space.shape, action_space)
        self.optimizer = torch.optim.RMSprop(self.policy.parameters(), self.learning_rate, eps=1e-5, alpha=0.99)

        self.trajectory_buffer = TrajectoryBuffer(
            self.n_steps,
            self.n_processes,
            self.observation_space.shape,
            self.action_space,
        )

    def update(self, trajectory):
        obs_shape = trajectory.obs.size()[2:]
        action_shape = trajectory.actions.size()[-1]

        action_log_probs = self.policy.evaluate_actions(
            trajectory.obs[:-1].view(-1, *obs_shape),
            trajectory.actions.view(-1, action_shape))

        log_probs = action_log_probs.view(self.n_steps, self.n_processes, 1)

        advantages = trajectory.returns[:-1] - trajectory.returns[:-1].mean()

        policy_loss = -(advantages.detach() * log_probs).mean()

        self.optimizer.zero_grad()
        policy_loss.backward()
        nn.utils.clip_grad_norm_(self.policy.parameters(),
                                 self.max_grad_norm)
        self.optimizer.step()

        return policy_loss.item()

    def save(self, filename):
        torch.save(self.policy.state_dict(), filename + "reinforce")
        torch.save(self.optimizer.state_dict(), filename + "_reinforce_optimizer")

    def load(self, filename):
        self.policy.load_state_dict(torch.load(filename + "_reinforce"))
        self.optimizer.load_state_dict(torch.load(filename + "_reinforce_optimizer"))

    # Runs policy for X episodes and returns average reward
    # A fixed seed is used for the eval environment
    def eval_policy(self, env_name, seed, eval_episodes=10):
        eval_env = gym.make(env_name)
        eval_env.seed(seed + 100)

        avg_reward = 0.
        for _ in range(eval_episodes):
            state, done = eval_env.reset(), False
            state = torch.FloatTensor([state])
            while not done:
                with torch.no_grad():
                    action, action_log_prob = self.policy.select_actions(state)
                state, reward, done, _ = eval_env.step(action)
                state = torch.FloatTensor([state])
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

        # Evaluate untrained policy
        evaluations = [self.eval_policy(args.env, args.seed)]

        # Set up state and rollout buffer
        state, done = env.reset()
        self.trajectory_buffer.obs[0].copy_(state)
        self.trajectory_buffer.to(device)

        # Initialize log information
        episode_reward = 0
        episode_timesteps = 0
        episode_num = 0
        t = 0

        # Initialize data list for plot
        x_timesteps = []
        y_rewards = []
        episode_y_rewards = torch.zeros((self.n_processes, 1))
        while t < args.max_timesteps:
            t += self.n_processes * self.n_steps
            episode_timesteps += 1

            # run environment n steps to collect rollout data
            for i in range(self.n_steps):
                # Sample actions
                with torch.no_grad():
                    action, action_log_prob = self.policy.select_actions(
                        self.trajectory_buffer.obs[i])
                next_state, reward, done, infos = env.step(action)

                # Use masks and bad masks to identify which represents true terminate
                masks = torch.FloatTensor(
                    [[0.0] if done_ else [1.0] for done_ in done])
                bad_masks = torch.FloatTensor(
                    [[0.0] if 'bad_transition' in info.keys() else [1.0]
                     for info in infos])

                self.trajectory_buffer.insert(next_state, action, action_log_prob, reward, masks, bad_masks)

                # Compute log and plot info
                episode_reward += sum(reward)[0]
                episode_y_rewards += reward

            # Use n steps rollout data to update actor-critic network
            self.trajectory_buffer.compute_returns(self.discount)
            self.update(self.trajectory_buffer)
            # Clear rollout buffer
            self.trajectory_buffer.after_update()

            if done[0]:
                # Print log
                print(
                    f"Total T: {t}/{args.max_timesteps} "
                    f"Episode Num: {episode_num + self.n_steps} "
                    f"Reward: {episode_reward/self.n_processes:.3f}")
                episode_reward = 0
                episode_timesteps = 0
                episode_num += self.n_processes

                # Reset environment
                state, done = env.reset()

                # Compute and reset plot info
                y_rewards.extend(episode_y_rewards.squeeze().tolist())
                episode_y_rewards = torch.zeros((self.n_processes, 1))
                x_timesteps.extend([i for i in range(t - self.n_processes * self.max_episode_timesteps,
                                                                  t, self.max_episode_timesteps)])


            # Evaluate episode
            if t % args.eval_freq == 0:
                evaluations.append(self.eval_policy(args.env, args.seed))
                np.save(f"./results/{file_name}", evaluations)
                if args.save_model: self.save(f"./models/{file_name}")

        return x_timesteps, y_rewards
