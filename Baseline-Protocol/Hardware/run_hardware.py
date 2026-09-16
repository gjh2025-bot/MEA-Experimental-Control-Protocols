"""
Baseline closed loop on the real MEA2100-Mini-60.

Each game step:
  1. Look at the pole.
  2. Stimulate: predictable if it's balanced, unpredictable if not.
  3. Listen to the neurons while the stimulation plays.
  4. Turn spike counts into an action (left or right).
  5. Step CartPole with that action.

Run (from this folder, MEA plugged in over USB):  python run_hardware.py
"""
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt

from feedback import is_balanced, lean_side, choose_action
from mea_device import MEADevice

EPISODES = 20
WINDOW_MS = 100   # how long each stimulate + listen step lasts
RENDER = False    # True opens a window showing the cart


def main():
    env = gym.make("CartPole-v1", render_mode="human" if RENDER else None)
    mea = MEADevice()
    mea.connect()

    episode_lengths = []
    try:
        for episode in range(EPISODES):
            observation, _ = env.reset()
            steps, done = 0, False

            while not done:
                pole_angle = observation[2]

                if is_balanced(pole_angle):
                    mea.stimulate_predictable(lean_side(pole_angle), WINDOW_MS)
                else:
                    mea.stimulate_unpredictable(WINDOW_MS)

                spike_counts = mea.read_spike_counts(WINDOW_MS)
                action = choose_action(spike_counts)

                observation, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                steps += 1

            episode_lengths.append(steps)
            print(f"Episode {episode + 1}/{EPISODES}: {steps} steps")
    finally:
        mea.close()  # always release the device, even after an error or Ctrl+C
        env.close()

    np.savetxt("episode_lengths.csv", episode_lengths, fmt="%d", header="steps", comments="")
    plt.plot(range(1, len(episode_lengths) + 1), episode_lengths, marker="o")
    plt.xlabel("Episode")
    plt.ylabel("Steps balanced")
    plt.title("Hardware baseline")
    plt.show()


if __name__ == "__main__":
    main()
