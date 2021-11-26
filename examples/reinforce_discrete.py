"""
Basic example of training a policy gradient with the REINFORCE algorithm in a gym environment

Code is heavily based on (or, in some places, completely taken from) the following article:
https://medium.com/@thechrisyoon/deriving-policy-gradients-and-implementing-reinforce-f887949bd63
"""

import gym
import numpy as np
import torch
from torch import nn
from tqdm import tqdm

class Policy(nn.Module):
    def __init__(self, observation_space, action_space):
        super().__init__()

        self.observation_space = observation_space
        self.action_space = action_space

        self.policy_network = self.build_network().cuda()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=3e-4)

    def get_action(self, observation):
        observation = torch.from_numpy(observation).float().unsqueeze(0).to(device)          # Convert to tensor and reshape
        probs = self.forward(observation)                                                    # Probability of taking each action, a
        action = np.random.choice(self.action_space.n, p=np.squeeze(probs.detach().cpu().numpy())) # Sample a random action
        log_prob = torch.log(probs.squeeze(0)[action]).to(device)                            # Select the log probability of taking that action pi_theta(a_t | s_t)
        return action, log_prob

    def build_network(self):
        return torch.nn.Sequential(
            nn.Linear(self.observation_space.shape[0], 64),
            nn.ReLU(),
            nn.Linear(64, self.action_space.n),
            nn.Softmax()
        )

    def forward(self, observation):
        return self.policy_network(observation)



def update_policy(policy: Policy, rewards, log_probs, gamma=0.9):
    discounted_rewards = []

    reward_to_go = 0
    for r in reversed(rewards):
        reward_to_go = r + gamma*reward_to_go
        discounted_rewards.append(reward_to_go)
    discounted_rewards = list(reversed(discounted_rewards))
    
    discounted_rewards = torch.tensor(discounted_rewards) # Convert to tensor
    discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-12) # normalize data

    policy_gradient = []
    for log_prob, reward_to_go in zip(log_probs, discounted_rewards):
        policy_gradient.append(-log_prob * reward_to_go)

    policy.optimizer.zero_grad()
    policy_gradient = torch.stack(policy_gradient).sum()
    policy_gradient.backward()
    policy.optimizer.step()


def test_policy(policy, env, max_steps=1000):
    observation = env.reset()
    for step in range(max_steps):
        env.render()
        action, _ = policy.get_action(observation)
        observation, _, done, _ = env.step(action)
        if done:
            break
    env.render()

if __name__ == '__main__':
    # Select GPU if possible
    device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")

    env = gym.make('CartPole-v0')

    policy = Policy(env.observation_space, env.action_space)

    max_episodes = 5000
    max_steps = 10000
    all_rewards = []

    # Train policy
    for episode in tqdm(range(max_episodes), desc="Training agent"):
        state = env.reset()
        log_probs = []
        rewards = []

        for step in range(max_steps):
            action, log_prob = policy.get_action(state)
            state, reward, done, _ = env.step(action)

            log_probs.append(log_prob)
            rewards.append(reward)

            if done:
                update_policy(policy, rewards, log_probs)
                break

        # Visualize policy
        if episode % 50 == 0:
            test_policy(policy, env, max_steps=max_steps)

    input("Training finished, press enter to see final model")
    test_policy(policy, env, max_steps=max_steps * 5)


    
    