import os
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import torch

# Path hack as we don't need this to be an installable package
root_path = os.path.dirname(
    os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )  # ant-v3-exp1  # experiments
)  # mujoco-2.1-rl-project
sys.path.insert(0, root_path)

from rl.algorithms.a2c_continuous import ActorCritic


if __name__ == "__main__":
    import gym
    import matplotlib.pyplot as plt

    model_dir = Path(__file__).parent / "models"

    env = gym.make("Ant-v3", healthy_z_range=(0.30, 2.0)).unwrapped

    a2c = ActorCritic(
        observation_space=env.observation_space,
        action_space=env.action_space,
        model_dir=model_dir,
        hidden_sizes=(256, 256),
        device="cuda:0",
    )
    training_stats = a2c.train(env, steps=500_000)

    plt.figure(figsize=(12, 8))
    for idx, (title, data) in enumerate(training_stats.items(), start=1):
        plt.subplot(1, 3, idx)
        plt.plot(data)
        plt.title(title)

    plt.savefig("a2c-500k-Ant-v3.png")
    model_path = model_dir / "a2c-ant-v3-final.pt"
    torch.save(a2c.state_dict(), model_path)

    while input("Enter 'quit' to quit:") != "quit":
        a2c.test(env, render=True, test_iterations=1)
