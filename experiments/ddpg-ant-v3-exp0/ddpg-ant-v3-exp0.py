import os
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import torch

# Path hack as we don't need this to be an installable package
root_path = os.path.dirname(
    os.path.dirname(os.path.dirname(__file__))  # ant-v3-exp1  # experiments
)  # mujoco-2.1-rl-project
sys.path.insert(0, root_path)

from rl.algorithms.ddpg import DDPG

if __name__ == "__main__":
    import gym

    model_dir = Path(__file__).parent / "models"

    env = gym.make("Ant-v3", healthy_z_range=(0.30, 2.0)).unwrapped

    ddpg = DDPG(
        observation_space=env.observation_space,
        action_space=env.action_space,
        hidden_sizes=(256, 256),
        update_freq=64,
        batch_size=128,
        update_threshold=4096,
        noise_std=0.1,
        model_dir=model_dir,
        device="cuda:0",
    )
    training_stats = ddpg.train(env, steps=500_000, render_freq=100_000)

    while input("Type 'quit' to quit: ") != "quit":
        ddpg.test(env, render=True)

    plt.figure(figsize=(12, 8))
    for idx, (title, data) in enumerate(training_stats.items(), start=1):
        plt.subplot(1, 4, idx)
        plt.plot(data)
        plt.title(title)

    plt.savefig("ddpg-500k-Ant-v3.png")
    model_path = model_dir / "ddpg-antv3-final.pt"
    torch.save(ddpg.state_dict(), model_path)

