import gym
import torch
from torch import nn
from tqdm import tqdm
import torch.distributions

class Actor(nn.Module):
    """
    Actor for an Actor-Critic learning algorithm

    Predicts and updates the agent's policy
    """
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.policy = self.initialize_policy_net(input_size, hidden_size, output_size)

        self.policy_history = torch.Tensor()
        self.episode_rewards = []

    def initialize_policy_net(self, input_size, hidden_size, output_size):
        return nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
            nn.Softmax(dim=1)
        )

    def forward(self, observation):
        observation = torch.Tensor(observation).float().unsqueeze(0)
        return self.policy(observation)

    def get_action(self, policy_probs):
        pi = torch.distributions.Categorical(policy_probs)
        action = pi.sample()

        # Append pi_theta(a | s) to history
        self.policy_history = torch.cat((self.policy_history, pi.log_prob(action)))
        return action

    def clear_buffers(self):
        self.policy_history = torch.Tensor()
        self.episode_rewards = []




class Critic(nn.Module):
    """
    Critic for an Actor-Critic learning algorithm

    Predicts and updates the value approximator function
    """
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.value_approximator = self.initialize_value_approximator(input_size, hidden_size)
        self.value_history = torch.Tensor()

    def initialize_value_approximator(self, input_size, hidden_size):
        return nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, observation):
        observation = torch.Tensor(observation).float()
        value = self.value_approximator(observation)
        self.value_history = torch.cat((self.value_history, value))
        return value

    def clear_buffers(self):
        self.value_history = torch.Tensor()


class AdvantageActorCritic(nn.Module):
    """
    Uses Actor-Critic learning algorithm to predict the value function an policy function

    Uses Advantage function ( A(s, a) = Q(s, a) - V(s) ) to determine gradients
    """
    def __init__(self, input_size, hidden_size, output_size, lr=3e-4, gamma=0.99):
        super(AdvantageActorCritic, self).__init__()
        super(AdvantageActorCritic, self).add_module(
            'actor', 
            Actor(input_size, hidden_size, output_size)
        )
        super(AdvantageActorCritic, self).add_module(
            'critic', 
            Critic(input_size, hidden_size)
        )
        
        self.gamma = gamma
        self.optim = torch.optim.Adam(self.parameters(), lr=lr)
        

    def forward(self, observation):
        value = self.critic(observation)
        policy = self.actor(observation)
        return value, policy

    def update(self):
        reward_to_go = 0
        rewards = []

        for reward in reversed(self.actor.episode_rewards):
            reward_to_go = reward + (self.gamma * reward_to_go)
            rewards.insert(0, reward_to_go)
        
        q_t = torch.Tensor(rewards).float()
        advantage = q_t - self.critic.value_history # Calculate the advantage
                                                    # Q(s, a) - V(s)

        # Calculate losses
        critic_loss = advantage.pow(2).mean() # Use a Mean Squared Error. 
                                              # In optimal case advantage is zero, so no subtraction is needed.

        actor_loss = (-self.actor.policy_history * advantage).mean() # ∇_θ(𝓙(π_θ)) = 𝔼[∇_θ(log π_θ(a_t | s_t) * A(s_t, a_t))]
                                                                     # Multiply by -1 for gradient descent as we want to maximize the expected return, 𝓙(π_θ).

        # Use gradient descent
        self.optim.zero_grad()
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
        action = a2c.actor.get_action(policy)
        observation, _, done, _ = env.step(action.numpy().squeeze(0))
        if done:
            break
    env.render()
    a2c.clear_buffers()



if __name__ == '__main__':
    max_episodes = 5000
    max_steps = 5000
    
    env = gym.make('CartPole-v0')
    
    a2c = AdvantageActorCritic(
        input_size=env.observation_space.shape[0], 
        hidden_size=64, 
        output_size=env.action_space.n
    )

    for episode in tqdm(range(max_episodes), desc="Training agent"):
        state = env.reset()
        rewards = []

        for step in range(max_steps):
            value, policy = a2c(state)
            action = a2c.actor.get_action(policy)

            state, reward, done, _ = env.step(action.numpy().squeeze(0))
            a2c.actor.episode_rewards.append(reward)

            if done:
                a2c.update()
                break

        if episode % 50 == 0:
            test_policy(a2c, env, max_steps)

    
        


    
    