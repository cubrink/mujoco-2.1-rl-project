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


if __name__ == "__main__":
    model_dir = Path(__file__).parent / "models"

    env = gym.make("Humanoid-v3",)
    env._max_episode_steps = 500
    sac = SAC(
        env,
        hidden_sizes=(256, 256),
        update_freq=64,
        num_update=64,
        update_threshold=4096,
        batch_size=128,
        model_dir=model_dir,
        alpha=0.5,
        device="cuda:0",
    )

    models_to_test = ["sac-2514-240000.pt", "sac-5396-260000.pt"]

    sac.load_state_dict(torch.load(model_dir / models_to_test[0]))

    while input("Type 'quit' to quit: ") != "quit":
        sac.test(render=True, test_iterations=1)
