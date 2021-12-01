import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributions as distributions

from tqdm import tqdm
from pathlib import Path
from copy import deepcopy
from rl.utils.misc import as_numpy, init_mlp, as_tensor, get_device
from rl.utils.buffers import FifoBuffer


class PiActor(nn.Module):
    """
    Actor for learning stochastic policies
    """

    LOG_STD_MAX = 2  # tanh(e^2) ~= 1.0
    LOG_STD_MIN = -20  # tanh(exp(-20)) ~= 0.0

    def __init__(self, network_sizes, device=None):
        super().__init__()
        self.device = get_device(device=device)
        self.mu = init_mlp(network_sizes)  # Mean of normal distribution
        self.log_sigma = init_mlp(network_sizes)  # Log Standard deviation
        self.to(self.device)

    def forward(self, state) -> distributions.Normal:
        """
        Returns the policy for a given observation
        """
        state = as_tensor(state, device=self.device)
        loc = self.mu(state)
        log_scale = self.log_sigma(state).clamp(self.LOG_STD_MIN, self.LOG_STD_MAX)
        scale = torch.exp(log_scale)
        policy = distributions.Normal(loc, scale)
        return policy


class DoubleQCritic(nn.Module):
    """
    Critic for learning state action value function, Q
    """

    def __init__(self, network_sizes, device=None):
        super().__init__()
        self.device = get_device(device=device)
        self._q1 = init_mlp(network_sizes)
        self._q2 = init_mlp(network_sizes)
        self.to(self.device)

    def _get_state_action(self, state, action):
        state = as_tensor(state, device=self.device)
        action = as_tensor(action, device=self.device)

        s_a = torch.cat((state, action), axis=-1)
        if s_a.dim() == 1:
            s_a = s_a.unsqueeze(0)
        return s_a

    def forward(self, state, action):
        """
        Returns the expected value of a state action pair
        """
        s_a = self._get_state_action(state, action)
        q1q2 = torch.stack(
            (self._q1(s_a).squeeze(-1), self._q2(s_a).squeeze(-1))
        )  # In the form [[q1_0, q1_1, ...], [q2_0, q2_1, ...]]
        return q1q2

    def q1(self, state, action):
        s_a = self._get_state_action(state, action)
        return self._q1(s_a).squeeze(1)

    def q2(self, state, action):
        s_a = self._get_state_action(state, action)
        return self._q2(s_a).squeeze(1)


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
        self.to(self.device)


class SAC(nn.Module):
    """
    Soft Actor Critic algorithm using PyTorch
    """

    def __init__(
        self,
        env,
        hidden_sizes,
        update_freq,
        update_threshold,
        batch_size=128,
        buffer_size=int(3e5),
        num_update=1,
        lr=5e-4,
        alpha=0.2,
        gamma=0.99,
        tau=0.995,
        alpha_decay=1,
        stationary_penalty=1,
        model_dir="./models",
        device=None,
    ):
        super().__init__()
        # Misc
        self.env = env
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self.device = get_device(device=device)
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True, parents=True)
        self.mse_threshold = 1e-4  # Threshold to prevent ant from standing still and gaining healthy rewards

        # Hyperparameters
        self.alpha = alpha
        self.gamma = gamma
        self.tau = tau
        self.lr = lr
        self.stationary_penalty = stationary_penalty
        self.alpha_decay = alpha_decay
        self.batch_size = batch_size
        self.num_update = num_update
        self.update_threshold = update_threshold
        self.update_freq = update_freq

        # Create replay buffer to hold environment transitions
        self.replay_buffer = FifoBuffer(
            self.observation_space.shape[0], self.action_space.shape[0], buffer_size
        )

        # Create network and target network
        self.ac = DoubleQActorCritic(
            self.observation_space, self.action_space, hidden_sizes, device=self.device
        )
        self.ac_target = deepcopy(self.ac)

        # Disable gradient in target network, it will be updated with
        # polyak averaging instead
        for p in self.ac_target.parameters():
            p.requires_grad = False

        # Define optimizers for the different networks being updated
        # with gradient descent
        self.optim_pi = optim.Adam(self.ac.actor.parameters(), lr=lr)
        self.optim_q1 = optim.Adam(self.ac.critic._q1.parameters(), lr=lr)
        self.optim_q2 = optim.Adam(self.ac.critic._q2.parameters(), lr=lr)

        self.to(self.device)

    def get_action(self, state, sample_from_dist=True):
        """
        Get action from the policy

        state: State of the environment
        sampel_from_dist: Selects behavior of selecting action from the policy.
            If True, a normal distribution is created and an action is sampled from the policy.
            If False, the mean value from the policy is used as the action. Use during test time.
        """
        policy = self.get_policy(state)
        if sample_from_dist:
            u = policy.sample()  # Sample from unsquashed distribution
        else:
            u = policy.loc
        action = u.tanh()
        return as_numpy(action)

    def get_policy(self, state) -> distributions.Normal:
        state = as_tensor(state, device=self.device)
        policy = self.ac.actor(state)
        return policy

    def update(self):
        data = self.replay_buffer.sample(self.batch_size)
        data = {k: as_tensor(v, device=self.device) for k, v in data.items()}
        s, a, r, s2, d = data["s"], data["a"], data["r"], data["s_prime"], data["d"]

        # Calculate targets
        with torch.no_grad():
            policy_s2 = self.get_policy(s2)
            # The equation for log_prob_a2 comes from:
            #   https://github.com/openai/spinningup/blob/master/spinup/algos/pytorch/sac/core.py#L59
            #
            #   In essence, this is a numerically stable way of converting the log_prob of the original
            #   distribution to a log_prob of the squashed distribution.
            u = policy_s2.rsample()
            a2 = u.tanh()
            log_prob_a2 = policy_s2.log_prob(u).sum(axis=-1)
            log_prob_a2 -= (2 * (math.log(2) - u - F.softplus(-2 * u))).sum(axis=1)
            q_target = self.ac_target.critic(s2, a2).min(axis=0).values
            target = r + self.gamma * (1 - d) * (q_target - self.alpha * log_prob_a2)

        # Update q1
        self.optim_q1.zero_grad()
        q1_loss = ((self.ac.critic.q1(s, a) - target) ** 2).mean()
        q1_loss.backward()
        self.optim_q1.step()

        # Update q2
        self.optim_q2.zero_grad()
        q2_loss = ((self.ac.critic.q2(s, a) - target) ** 2).mean()
        q2_loss.backward()
        self.optim_q2.step()

        # Critic doesn't need gradient during actor updates
        # this speeds up training by ~20%
        for p in self.ac.critic.parameters():
            p.requires_grad = False

        # Update policy
        self.optim_pi.zero_grad()
        policy_s = self.get_policy(s)
        u_tilde = policy_s.rsample()
        a_tilde = u_tilde.tanh()
        log_prob_a_tilde = policy_s.log_prob(u_tilde).sum(axis=-1)
        log_prob_a_tilde -= (
            2 * (math.log(2) - u_tilde - F.softplus(-2 * u_tilde))
        ).sum(axis=1)
        pi_loss = (
            self.ac.critic(s, a_tilde).min(axis=0).values
            - self.alpha * log_prob_a_tilde
        ).mean()
        pi_loss *= -1  # We want to use gradient ascent, so *-1
        pi_loss.backward()
        self.optim_pi.step()

        for p in self.ac.critic.parameters():
            p.requires_grad = True

        # Update target networks with polyak averaging
        self.polyak_average(target=self.ac_target, source=self.ac)

        # Decay alpha
        self.alpha *= self.alpha_decay

        return q1_loss, q2_loss, pi_loss

    def train(
        self, steps, render_freq=None, random_before_threshold=True, max_steps=None,
    ):
        state_curr = self.env.reset()
        last_render = 0  # Last step that was rendered
        stats = {
            "reward_history": [],  # Cumulative rewards from episodes
            "q1_loss": [],
            "q2_loss": [],
            "pi_loss": [],
        }
        max_reward = -100
        render = False
        render_now = False
        quit_early = False  # Change this value in the debugger to exit early!

        for step in tqdm(range(steps)):
            # Get action to take in environment
            if random_before_threshold and step < self.update_threshold:
                action = self.action_space.sample()
            else:
                action = self.get_action(state_curr)

            # Take step in environment
            state_next, reward, done, log = self.env.step(action)

            # Store transition
            self.replay_buffer.store((state_curr, action, reward, state_next, done))

            # Update state
            if done:
                state_curr = self.env.reset()
            else:
                state_curr = state_next

            # Perform updates, if needed
            if (step >= self.batch_size) and (step % self.update_freq) == 0:
                for _ in range(self.num_update):
                    q1_loss, q2_loss, pi_loss = self.update()
                    stats["q1_loss"].append(q1_loss.item())
                    stats["q2_loss"].append(q2_loss.item())
                    stats["pi_loss"].append(pi_loss.item())

            # Do testing
            if step % 10000 == 0:
                # Determine whether to render
                render = True
                if render_freq and (step - last_render) >= render_freq:
                    last_render = step
                    render = True

                # Test agent
                test_result = self.test(render=render, max_steps=max_steps)
                stats["reward_history"].append(test_result)

                # Reset environment
                state_curr = self.env.reset()

                if test_result > max_reward:
                    max_reward = test_result
                    model_path = self.model_dir / f"sac-{int(test_result)}-{step}.pt"
                    torch.save(self.state_dict(), model_path)

            # Change these flags in the debugger
            if render_now:
                self.env.render()
            if quit_early:
                break

        return stats

    def test(self, env=None, max_steps=None, render=False, test_iterations=10):
        if env is None:
            env = self.env

        episode_rewards = []

        for episode in range(test_iterations):
            state_curr = env.reset()

            stationary_count_threshold = 50
            stationary_count = 0
            total_reward = 0

            if max_steps is None:
                max_steps = 25000

            for step in range(max_steps):
                action = self.get_action(state_curr, sample_from_dist=False)
                state_next, reward, done, _ = env.step(action)
                mse = ((state_curr - state_next) ** 2).sum()

                if mse < self.mse_threshold:
                    stationary_count += 1
                    # reward -= self.stationary_penalty
                if stationary_count >= stationary_count_threshold:
                    done = True

                state_curr = state_next
                total_reward += reward
                if render:
                    env.render()
                if done:
                    break

            episode_rewards.append(total_reward)
            render = False  # Only render the first iteration, if rendering
        mean_reward = sum(episode_rewards) / len(episode_rewards)
        return mean_reward

    @torch.no_grad()
    def polyak_average(self, target, source):
        for p_targ, p in zip(target.parameters(), source.parameters()):
            p_targ.data.mul_(self.tau)
            p_targ.data.add_((1 - self.tau) * p.data)

