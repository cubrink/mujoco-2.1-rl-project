from copy import deepcopy

from tqdm import tqdm
from torch import distributions
from utils.buffers import FifoBuffer
from utils.networks import init_mlp, as_tensor

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt


class MuActor(nn.Module):
    """
    Actor for learning deterministic policies
    """

    def __init__(self, network_sizes, device=None):
        super().__init__()
        self.mu = init_mlp(network_sizes, output_activation=nn.Tanh)
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, observation):
        observation = as_tensor(observation, device=self.device)
        return self.mu(observation)


class QCritic(nn.Module):
    """
    Critic for learning the state action value function, Q
    """

    def __init__(self, network_sizes, device=None):
        super().__init__()
        self.q = init_mlp(network_sizes)
        self.device = torch.device(
            device if (torch.cuda.is_available() and device) else "cpu"
        )
        self.to(self.device)

    def forward(self, state, action):
        """
        Returns the expected value of a state action pair
        """
        state = as_tensor(state, device=self.device)
        action = as_tensor(action, device=self.device)
        s_a = torch.cat((state, action), axis=-1)
        return self.q(s_a).squeeze(-1)


class ActorCritic(nn.Module):
    """
    Handle actor-critic interactions
    """

    def __init__(self, observation_space, action_space, hidden_sizes, device=None):
        super().__init__()
        self.observation_space = observation_space
        self.action_space = action_space
        self.device = torch.device(
            device if (torch.cuda.is_available() and device) else "cpu"
        )

        actor_sizes = (observation_space.shape[0], *hidden_sizes, action_space.shape[0])
        self.actor = MuActor(actor_sizes, device=self.device)
        super().add_module("actor", self.actor)

        critic_sizes = (
            observation_space.shape[0] + action_space.shape[0],
            *hidden_sizes,
            1,
        )
        self.critic = QCritic(critic_sizes, device=self.device)
        super().add_module("critic", self.critic)


class DDPG:
    """
    Deep Determinisitic Policy Gradient algorithm using PyTorch
    """

    def __init__(
        self,
        observation_space,
        action_space,
        hidden_sizes,
        update_freq,
        update_threshold,
        noise_std,
        batch_size=128,
        buffer_size=int(1e6),
        lr=1e-3,
        gamma=0.99,
        tau=0.995,
        device=None,
    ):
        # Misc
        self.observation_space = observation_space
        self.action_space = action_space
        self.device = torch.device(
            device if (torch.cuda.is_available() and device) else "cpu"
        )

        # Hyperparameters
        self.gamma = gamma
        self.tau = tau
        self.lr = lr
        self.noise_std = noise_std
        self.batch_size = batch_size
        self.update_freq = update_freq
        self.update_threshold = update_threshold

        self.replay_buffer = FifoBuffer(
            observation_size=observation_space.shape[0],
            action_size=action_space.shape[0],
            buffer_size=buffer_size,
        )

        # Create network and target network
        self.ac = ActorCritic(
            observation_space, action_space, hidden_sizes, device=device
        )
        self.ac_target = deepcopy(self.ac)

        # Disable gradient in target network, it will be updated with
        # polyak averaging instead
        for p in self.ac_target.parameters():
            p.requires_grad = False

        # Create optimizers for original network
        self.optim_actor = optim.Adam(self.ac.actor.parameters(), lr=lr)
        self.optim_critic = optim.Adam(self.ac.critic.parameters(), lr=lr)

    def update(self):
        data = self.replay_buffer.sample(self.batch_size)
        data = {k: as_tensor(v, device=self.device) for k, v in data.items()}
        s, a, r, s2, d = data["s"], data["a"], data["r"], data["s_prime"], data["d"]

        with torch.no_grad():
            # y(r, s', d) = r + γ*(1-d)*Q_target(s', μ_target(s'))
            target = r + self.gamma * (1 - d) * self.ac_target.critic(
                s2, self.ac_target.actor(s2)
            )

        # Update Q Value Network
        self.optim_critic.zero_grad()
        q = self.ac.critic(s, a)
        q_loss = nn.MSELoss(reduction="sum")(q, target)
        q_loss.backward()
        self.optim_critic.step()

        # Update Mu Policy Network
        self.optim_actor.zero_grad()
        mu_loss = (
            -1 * self.ac.critic(s, self.ac.actor(s)).mean()
        )  # -1 because we want to maximize Q wrt mu
        mu_loss.backward()
        self.optim_actor.step()

        # Update target networks with polyak averaging
        self.polyak_average(target=self.ac_target, source=self.ac)

        return q_loss, mu_loss

    def train(self, env, steps, render_freq=None, random_before_threshold=True):
        """
        Trains the network using the DDPG algorithm

        env: The environment to use following the OpenAI gym API
        steps: The amount of steps to train for
        render_freq: How many steps to wait before rendering the model for the user
        random_before_threshold: If True, before the step theshold is met, random actions are
            sampled from the action space instead of from the network
        """
        state_curr = env.reset()
        reward_history = []  # Track the cumulative rewards given per attempt in the env
        q_losses = []
        mu_losses = []
        attempt_rewards = (
            []
        )  # Hold reward at each timestep of the current attempt in the env
        last_render = 0  # Last step that was rendered
        for step in tqdm(range(steps)):
            # Get action to take in environment
            if random_before_threshold and step < self.update_threshold:
                action = self.action_space.sample()
            else:
                action = self.get_action(state_curr)

            # Take step in environment
            state_next, reward, done, _ = env.step(action)

            # Store transition
            self.replay_buffer.store((state_curr, action, reward, state_next, done))
            attempt_rewards.append(reward)

            # Reset environment, if needed
            if done:
                if render_freq and (step - last_render) >= render_freq:
                    last_render = step
                    self.render(env)

                state_curr = env.reset()
                reward_history.append(sum(attempt_rewards))
                attempt_rewards.clear()
            else:
                state_curr = state_next

            # Perform updates, if needed
            if (step >= self.update_threshold) and (step & self.update_freq) == 0:
                q_loss, mu_loss = self.update()
                q_losses.append(q_loss.item())
                mu_losses.append(mu_loss.item())

        # If rendering, show model before exiting
        if render_freq:
            self.render(env)

        return reward_history, q_losses, mu_losses

    def render(self, env, max_steps=1000):
        state = env.reset()
        total_reward = 0
        for step in range(max_steps):
            action = self.get_action(state, add_noise=False)
            state, reward, done, _ = env.step(action)
            total_reward += reward
            env.render()
            if done:
                break
        return total_reward

    @torch.no_grad()
    def polyak_average(self, target, source):
        for p_targ, p in zip(target.parameters(), source.parameters()):
            p_targ.data.mul_(self.tau)
            p_targ.data.add_((1 - self.tau) * p.data)

    def get_action(self, state, add_noise=True):
        state = as_tensor(state, device=self.device)
        action = self.ac.actor.mu(state)
        noise = 0
        if add_noise:
            noise = self.get_noise()
        return self.clamp_action(action + noise).cpu().detach().numpy()

    def get_noise(self):
        size = self.action_space.shape[0]
        dist = distributions.Normal(
            torch.zeros(size), torch.ones(size) * self.noise_std
        )
        return dist.sample().to(self.device)

    def clamp_action(self, action):
        min_action = self.action_space.low[0]
        max_action = self.action_space.high[0]
        return action.clamp(min_action, max_action)


if __name__ == "__main__":
    import gym
    from gym.wrappers import Monitor

    env = gym.make("Pendulum-v1")

    ddpg = DDPG(
        observation_space=env.observation_space,
        action_space=env.action_space,
        hidden_sizes=(128, 128),
        update_freq=64,
        update_threshold=1000,
        noise_std=0.1,
        device="cuda:0",
    )
    reward_history, q_loss, mu_loss = ddpg.train(env, steps=30_000, render_freq=100_000)

    while input("Type 'quit' to quit: ") != "quit":
        ddpg.render(env)

    plt.figure(figsize=(12, 8))
    plt.subplot(1, 3, 1)
    plt.plot(reward_history)
    plt.title("Reward history")

    plt.subplot(1, 3, 2)
    plt.plot(q_loss)
    plt.title("Q loss")

    plt.subplot(1, 3, 3)
    plt.plot(mu_loss)
    plt.title("μ Loss")

    plt.savefig("ddpg-30k-pendulum-solved.png")

    monitor = Monitor(env, "./video", force=True)
    ddpg.render(monitor)
