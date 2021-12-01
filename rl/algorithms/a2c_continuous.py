import gym
import torch
import torch.nn as nn
import torch.distributions as distributions
from tqdm import tqdm


class Actor(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super().__init__()
        log_std = -0.05 * torch.ones(output_size, dtype=torch.float32)
        self.log_std = torch.nn.Parameter(torch.as_tensor(log_std))
        self.mu_net = self.initialize_policy_net(input_size, hidden_size, output_size)
        self.buffers = {"logp_a_history": [], "episode_rewards": []}

    def initialize_policy_net(self, input_size, hidden_size, output_size):
        """
        Initializes a network that, given an observation, outputs the mean values for 
        the normal distribution that the action is sampled from
        """
        return nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, observation):
        policy = distributions.Normal(
            loc=self.mu_net(observation), scale=torch.exp(self.log_std)
        )
        return policy

    def get_action(self, policy):
        # Note: Values of logp_a are lower than I initially expected
        #       (with exp(logp_a) much less than .9-1.0). Recall that if you
        #       plot Normal(0,1) the peak probability density is about .40
        #       This is why the values are lower than I thought they would be
        action = policy.rsample()
        logp_a = policy.log_prob(action)
        self.buffers["logp_a_history"].append(logp_a)
        return action, logp_a

    def clear_buffers(self):
        self.buffers = {"logp_a_history": [], "episode_rewards": []}

    def __getitem__(self, key):
        return self.buffers[key]


class Critic(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.value_approximator = self.initialize_value_approximator(
            input_size, hidden_size
        )
        self.buffers = {"value_history": []}

    def initialize_value_approximator(self, input_size, hidden_size):
        return nn.Sequential(
            nn.Linear(input_size, hidden_size), nn.ReLU(), nn.Linear(hidden_size, 1)
        )

    def forward(self, observation):
        value = self.value_approximator(observation)
        self.buffers["value_history"].append(value)
        return value

    def clear_buffers(self):
        self.buffers = {"value_history": []}

    def __getitem__(self, key):
        return self.buffers[key]


class ActorCritic(nn.Module):
    def __init__(
        self, observation_space, action_space, hidden_size=64, lr=3e-4, gamma=0.99
    ):
        super().__init__()
        super(ActorCritic, self).add_module(
            "actor",
            Actor(observation_space.shape[0], action_space.shape[0], hidden_size),
        )
        super(ActorCritic, self).add_module(
            "critic", Critic(observation_space.shape[0], hidden_size)
        )
        self.gamma = gamma
        self.optim = torch.optim.Adam(self.parameters(), lr=lr)

    def forward(self, observation):
        observation = torch.Tensor(observation).float()
        value = self.critic(observation)
        policy = self.actor(observation)
        return value, policy

    def update(self):
        self.optim.zero_grad()

        reward_to_go = 0
        rewards = []

        # Calculate Extended Advantage Estimate
        for reward in reversed(self.actor["episode_rewards"]):
            reward_to_go = reward + (self.gamma * reward_to_go)
            rewards.insert(0, reward_to_go)

        # Convert values to tensors
        q_t = torch.Tensor(rewards)
        value_history = torch.Tensor(self.critic["value_history"])
        logp_a_history = torch.stack(self.actor["logp_a_history"])

        # Calcualte loss of each module
        advantage = q_t - value_history  # Q(s, a) - V(s)
        critic_loss = advantage.pow(
            2
        ).mean()  # Minimize advantage**2, acting optimally means that advantage should be 0
        actor_loss = (
            -logp_a_history.sum(axis=1) * advantage
        ).mean()  # ∇_θ(𝓙(π_θ)) = 𝔼[∇_θ(log π_θ(a_t | s_t) * A(s_t, a_t))]

        # Use gradient descent
        loss = critic_loss + actor_loss
        loss.backward()
        self.optim.step()

        # Reset buffers for actor/critic
        self.clear_buffers()
        return actor_loss, critic_loss

    def clear_buffers(self):
        self.actor.clear_buffers()
        self.critic.clear_buffers()


def test_policy(a2c, env, max_steps):
    observation = env.reset()
    for step in range(max_steps):
        env.render()
        value, policy = a2c(observation)
        action, logp_a = a2c.actor.get_action(policy)
        observation, _, done, _ = env.step(action.numpy())
        if done:
            break
    env.render()
    a2c.clear_buffers()


if __name__ == "__main__":
    max_episodes = 10000
    max_steps = 5000

    # env = gym.make('MountainCarContinuous-v0')
    env = gym.make("HalfCheetah-v3")

    a2c = ActorCritic(
        observation_space=env.observation_space, action_space=env.action_space,
    )

    for episode in tqdm(range(max_episodes), desc="Training agent"):
        state = env.reset()
        rewards = []

        for step in range(max_steps):
            value, policy = a2c(state)
            action, logp_a = a2c.actor.get_action(policy)

            state, reward, done, _ = env.step(action.numpy())
            a2c.actor["episode_rewards"].append(reward)

            if done:
                a2c.update()
                break

        # if episode % 200 == 0:
        #     test_policy(a2c, env, max_steps)

    while input("Enter 'quit' to exit") != "quit":
        test_policy(a2c, env, max_steps)
