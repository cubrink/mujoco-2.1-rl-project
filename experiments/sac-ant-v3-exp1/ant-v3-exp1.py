import os
import sys

# Path hack as we don't need this to be an installable package
root_path = os.path.dirname(
    os.path.dirname(os.path.dirname(__file__))  # ant-v3-exp1  # experiments
)  # mujoco-2.1-rl-project
sys.path.insert(0, root_path)

import torch
from rl.algorithms.sac import SAC

if __name__ == "__main__":
    import gym
    import matplotlib.pyplot as plt

    # This experiment was broken!
    # It used a custom reward function that looked like this:

    # state_next, reward, done, log = self.env.step(action)
    # distance_from_orig_next = log["x_position"]
    # distance_reward = max(
    #     (distance_from_orig_next - distance_from_orig_curr), 0
    # )
    # reward += distance_reward

    # But distance_from_orig_curr was never reset!
    print(
        "This was a broken experiment so it cannot be run again. The program will now exit."
    )
    quit()

    torch.set_num_threads(torch.get_num_threads())

    # env = gym.make("Pendulum-v1")
    env = gym.make(
        "Ant-v3",
        # healthy_z_range=(0.30, 2.0),
        healthy_reward=0,
        # ctrl_cost_weight=1e-2,
    ).unwrapped  # Crucial! This gets rid of the time limit so it doesn't stop at 1000 steps every time
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
    training_stats = sac.train(steps=1_000_000, render_freq=50_000, max_steps=10000)

    while input("Type 'quit' to quit: ") != "quit":
        sac.test(render=True)

    plt.figure(figsize=(12, 8))
    for idx, (title, data) in enumerate(training_stats.items(), start=1):
        plt.subplot(1, 4, idx)
        plt.plot(data)
        plt.title(title)

    plt.savefig("sac-1000k-ant-v3.png")
