import torch
import torch.nn as nn
import torch.distributions as distributions

from utils.misc import init_mlp, as_tensor, get_device


class PiActor(nn.Module):
    """
    Actor for learning stochastic policies
    """

    def __init__(self, network_sizes, device=None):
        super().__init__()
        self.device = get_device(device=device)
        self.mu_net = init_mlp(network_sizes)  # Mean of normal distribution
        self.log_sigma_net = init_mlp(network_sizes)  # Log Standard deviation
        self.to(self.device)

    def forward(self, state):
        """
        Returns the policy for a given observation
        """
        state = as_tensor(state, device=self.device)
        loc = self.mu_net(state)
        scale = torch.exp(self.log_sigma_net(state))
        policy = distributions.Normal(loc, scale)
        return policy


class DoubleQCritic(nn.Module):
    """
    Critic for learning state action value function, Q
    """

    def __init__(self, network_sizes, device=None):
        super().__init__()
        self.device = get_device(device=device)
        self.q1 = init_mlp(network_sizes)
        self.q2 = init_mlp(network_sizes)
        self.to(self.device)

    def forward(self, state, action):
        """
        Returns the expected value of a state action pair
        """
        state = as_tensor(state, device=self.device)
        action = as_tensor(action, device=self.device)

        # TODO: Remove this
        # double_up = lambda x: torch.stack((x, x))
        # state = double_up(state)
        # action = double_up(action)

        s_a = torch.cat((state, action), axis=-1)
        if s_a.dim() == 1:
            s_a = s_a.unsqueeze(0)
        return torch.stack(
            (self.q1(s_a).squeeze(-1), self.q2(s_a).squeeze(-1))
        )  # Return in the form [[q1_0, q1_1, ...], [q2_0, q2_1, ...]]


class DoubleQActorCritic(nn.Module):
    """
    Handles actor-critic interactions with two Q value functions to allow 
    using a clipped double-Q technique
    """

    def __init__(self, observation_space, action_space, hidden_sizes, device=None):
        super().__init__()
        self.observation_space = observation_space
        self.action_space = action_space
        self.device = get_device(device=device)

        actor_sizes = (observation_space.shape[0], *hidden_sizes, action_space.shape[0])
        self.actor = PiActor(actor_sizes, device=self.device)
        super().add_module("actor", self.actor)

        critic_sizes = (
            observation_space.shape[0] + action_space.shape[0],
            *hidden_sizes,
            1,
        )
        self.critic = DoubleQCritic(critic_sizes, device=self.device)
        super().add_module("critic", self.critic)


if __name__ == "__main__":
    import gym

    env = gym.make("Pendulum-v1")

    observation_space = env.observation_space
    action_space = env.action_space
    hidden_sizes = (128,)
    critic_sizes = (
        observation_space.shape[0] + action_space.shape[0],
        *hidden_sizes,
        1,
    )

    critic = DoubleQCritic(critic_sizes)

    state = env.reset()
    action = action_space.sample()

    result = critic(state, action)

    print(result)
