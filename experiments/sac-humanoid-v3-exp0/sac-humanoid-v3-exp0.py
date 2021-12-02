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
from rl.algorithms.sac import SAC
import matplotlib.pyplot as plt


if __name__ == "__main__":
    torch.set_num_threads(torch.get_num_threads())
    model_dir = Path(__file__).parent / "models"

    env = gym.make(
        "Humanoid-v3",
    ).unwrapped  # Crucial! This gets rid of the time limit so it doesn't stop at 1000 steps every time

    sac = SAC(
        env,
        hidden_sizes=(256, 256),
        update_freq=64,
        num_update=64,
        update_threshold=4096,
        batch_size=128,
        alpha=0.5,
        model_dir=model_dir,
        device="cuda:0",
    )
    training_stats = sac.train(steps=1_250_000, render_freq=50_000, max_steps=10000)

    while input("Type 'quit' to quit: ") != "quit":
        sac.test(render=True)

    plt.figure(figsize=(12, 8))
    for idx, (title, data) in enumerate(training_stats.items(), start=1):
        plt.subplot(1, 4, idx)
        plt.plot(data)
        plt.title(title)

    plt.savefig("sac-1250k-humanoid-v3.png")
