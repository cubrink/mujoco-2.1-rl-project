import gym
import numpy as np

env = gym.make('Ant-v2')
obs = env.reset()
obs_dim = env.observation_space.shape[0]
act_dim = env.action_space.shape[0]

print (obs_dim,act_dim)

for i in range(1000):
    env.render()
    action = np.random.randn(act_dim,1)
    action = action.reshape((1,-1)).astype(np.float32)
    obs, reward, done, _ = env.step(np.squeeze(action, axis=0))