import os
import sys

from pathlib import Path

# Path hack as we don't need this to be an installable package
root_path = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # ant-v3-exp1  # experiments
)  # mujoco-2.1-rl-project
sys.path.insert(0, root_path)

import torch
from rl.algorithms.sac import SAC


if __name__ == "__main__":
    import gym
    
    env = gym.make(
        "Ant-v3",
        healthy_z_range=(0.30, 2.0),
        # healthy_reward=0.01,
        # ctrl_cost_weight=1e-2,
    ).unwrapped  # Crucial! This gets rid of the time limit so it doesn't stop at 1000 steps every time
    # env._max_episode_steps = 25_000
    sac = SAC(
        env,
        hidden_sizes=(256, 256),
        update_freq=64,
        num_update=64,
        update_threshold=4096,
        batch_size=128,
        alpha=0.5,
        device="cuda:0",
    )
    
    sac.load_state_dict(torch.load(Path("models/sac-32057-478749.pt")))
    
    while input("Type 'quit' to quit: ") != "quit":
        sac.test(render=True)

    plt.figure(figsize=(12, 8))
    for idx, (title, data) in enumerate(training_stats.items(), start=1):
        plt.subplot(1, 4, idx)
        plt.plot(data)
        plt.title(title)
