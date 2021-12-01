import os
import sys
from pathlib import Path

# Path hack as we don't need this to be an installable package
root_path = os.path.dirname(
    os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )  # ant-v3-exp1  # experiments
)  # mujoco-2.1-rl-project
sys.path.insert(0, root_path)

import gym
import torch
from rl.algorithms.a2c_continuous import ActorCritic

if __name__ == "__main__":
    model_dir = Path(__file__).parent / "models"

    env = gym.make("Ant-v3", healthy_z_range=(0.3, 2.0))
    env._max_episode_steps = 10_000

    a2c = ActorCritic(
        observation_space=env.observation_space,
        action_space=env.action_space,
        model_dir=model_dir,
        hidden_sizes=(256, 256),
        device="cuda:0",
    )

    a2c.load_state_dict(torch.load(model_dir / "a2c-ant-v3-final.pt"))

    while input("Enter 'quit' to quit:") != "quit":
        a2c.test(env, render=True, test_iterations=1, sleep_time=0.02)

