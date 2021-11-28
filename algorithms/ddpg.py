from gym.core import ObservationWrapper
from numpy import isin
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributions as distributions

from tqdm import tqdm
from copy import deepcopy

from utils.buffers import FifoBuffer


class Actor(nn.Module):
    """
    Controls learning of the policy
    """

    def __init__(
        self, observation_space, action_space, hidden_size, lr=3e-4, device=None
    ):
        super().__init__()
        self.mu = self._init_mu_net(
            observation_space.shape[0], hidden_size, action_space.shape[0]
        )
        self.mu_targ = self._create_targ_net(self.mu)
        self.optim = optim.Adam(self.mu.parameters(), lr=lr)
        self.device = device
        self.to(device)

    def forward(self, observation):
        if not isinstance(observation, torch.Tensor):
            observation = torch.Tensor(observation).to(self.device)
        else:
            observation = observation.to(self.device)
        return self.mu(observation)

    def _init_mu_net(self, input_size, hidden_size, output_size):
        return nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
        )

    def _create_targ_net(self, net):
        targ_net = deepcopy(net)
        for p in targ_net.parameters():
            p.requires_grad = False
        return targ_net


class Critic(nn.Module):
    """
    Controls learning the value function
    """

    def __init__(
        self, observation_space, action_space, hidden_size, lr=3e-4, device=None
    ):
        super().__init__()
        input_size = observation_space.shape[0] + action_space.shape[0]
        self.q = self._init_q_net(input_size, hidden_size)
        self.q_targ = self._create_targ_net(self.q)
        self.optim = optim.Adam(self.q.parameters(), lr=lr)
        self.device = device
        self.to(device)

    def forward(self, observation, action):
        if not isinstance(observation, torch.Tensor):
            observation = torch.Tensor(observation).to(self.device)
        else:
            observation = observation.to(self.device)
        if not isinstance(observation, torch.Tensor):
            action = torch.Tensor(action).to(self.device)
        else:
            action = action.to(self.device)

        return self.q(torch.cat((observation, action), axis=1)).squeeze(1)

    def _init_q_net(self, input_size, hidden_size):
        return nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def _create_targ_net(self, net):
        targ_net = deepcopy(net)
        for p in targ_net.parameters():
            p.requires_grad = False
        return targ_net


class DDPGActorCritic(nn.Module):
    """
    Controls the Actor-Critic interaction for DDPG
    """

    def __init__(
        self,
        observation_space,
        action_space,
        buffer_size,
        batch_size,
        gamma,
        polyak,
        lr,
        noise_std,
        update_freq,
        update_threshold=512,
        hidden_size=128,
        device=None,
    ):
        super().__init__()
        self.observation_space = observation_space
        self.action_space = action_space
        self.gamma = gamma
        self.polyak = polyak
        self.lr = lr
        self.update_freq = update_freq
        self.update_threshold = update_threshold
        self.batch_size = batch_size
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        self.actor = Actor(
            observation_space, action_space, hidden_size, lr=lr, device=self.device
        )  # Policy approximator
        self.critic = Critic(
            observation_space, action_space, hidden_size, lr=lr, device=self.device
        )  # Value approximator
        self.replay_buffer = FifoBuffer(
            observation_size=observation_space.shape[0],
            action_size=action_space.shape[0],
            buffer_size=buffer_size,
        )
        self.noise_generator = distributions.Normal(0, noise_std)
        self.to(self.device)

    def get_action(self, observation, add_noise=True):
        observation = self._as_tensor(observation)
        noise = 0
        if add_noise:
            noise = self.noise_generator.sample()
        action = self.actor(observation) + noise
        action = action.clamp(self.action_space.low[0], self.action_space.high[0])
        return action.cpu().detach().numpy()

    def train(self, env, total_steps, render_freq=None):
        state_curr = env.reset()
        reward_history = []
        attempt_rewards = []
        for step in tqdm(range(total_steps)):
            if step < self.update_threshold:
                action = self.action_space.sample()
            else:
                action = self.get_action(state_curr, add_noise=True)
            state_next, reward, done, _ = env.step(action)
            self.replay_buffer.store((state_curr, action, reward, state_next, done))

            if step >= self.update_threshold and (step % self.update_freq) == 0:
                for _ in range(self.update_freq):
                    self.update()

            if done:
                state_curr = env.reset()
                reward_history.append(sum(attempt_rewards))
                attempt_rewards = []
            else:
                state_curr = state_next
                attempt_rewards.append(reward)

            if render_freq and step % render_freq == 0:
                self.test(env)
                attempt_rewards = []

        return reward_history

    def update(self):
        # Sample random transitions
        data = self.replay_buffer.sample(batch_size=self.batch_size)
        data = {k: torch.Tensor(v) for k, v in data.items()}

        # Calculate the target value
        with torch.no_grad():
            mu_targ_actions = self.actor.mu_targ(
                data["s_prime"]
            )  # Get actions from mu_targ policy
            q_targ_inputs = torch.cat(
                (data["s_prime"], mu_targ_actions), axis=1
            )  # Prepare inputs for q_targ
            target = data["r"] + self.gamma * (1 - data["d"]) * self.critic.q_targ(
                q_targ_inputs
            ).squeeze(1)

        # Update the q funtion
        self.critic.optim.zero_grad()
        q = self.critic(data["s"], data["a"])
        q_loss = ((q - target) ** 2).mean()
        q_loss.backward()
        self.critic.optim.step()

        # Update the policy
        self.actor.optim.zero_grad()
        mu_loss = (-self.critic(data["s"], self.actor.mu(data["s"]))).mean()
        mu_loss.backward()
        self.actor.optim.step()

        # Update targ networks with polyak averaging
        self._polyak_average(self.actor.mu_targ, self.actor.mu)
        self._polyak_average(self.critic.q_targ, self.critic.q)

    @torch.no_grad()
    def test(self, env, max_steps=1000):
        state = env.reset()
        env.render()
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
    def _polyak_average(self, net, other):
        for p, p_targ in zip(other.parameters(), net.parameters()):
            # Polyak averaging: p_targ = ρ*targ + (1 - ρ)*p
            p_targ.data.mul_(self.polyak)
            p_targ.data.add_((1 - self.polyak) * p.data)

    def _as_tensor(self, arr):
        """Quickly convert array to tensor"""
        return torch.Tensor(arr).to(self.device)


if __name__ == "__main__":
    import gym
    import matplotlib.pyplot as plt

    env = gym.make("Pendulum-v1")
    state = env.reset()
    action = env.action_space.sample()
    ddpg = DDPGActorCritic(
        observation_space=env.observation_space,
        action_space=env.action_space,
        buffer_size=100000,
        batch_size=128,
        gamma=0.99,
        polyak=0.995,
        lr=1e-3,
        noise_std=0.05,
        update_freq=64,
        device="cpu",
    )
    reward_history = ddpg.train(env, total_steps=200_000, render_freq=25000)

    plt.plot(reward_history)
    plt.show()

    while input("Type 'quit' to quit: ") != "quit":
        ddpg.test(env)

