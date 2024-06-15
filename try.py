from __future__ import annotations

from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.patches import Patch

import gymnasium as gym


# Let's start by creating the blackjack environment.
# Note: We are going to follow the rules from Sutton & Barto.
# Other versions of the game can be found below for you to experiment.

env = gym.make("Blackjack-v1", sab=False, natural=False, render_mode="rgb_array")

terminated = False
observation, info = env.reset()

# Let's play a game of blackjack.
while not terminated:
    print(f"Player's hand: {observation[0]}")
    print(f"Dealer's hand: {observation[1]}")
    print(f"Usable ace: {observation[2]}")
    print(f"Done: {terminated}\n")
    image = env.render()
    plt.imshow(image)
    plt.show()

    action = int(input("Choose an action (0: stick, 1: hit): "))
    observation, reward, terminated, truncated, info = env.step(action)
    print(f"Observation: {observation}")
    print(f"Reward: {reward}\n")
    image = env.render()
    plt.imshow(image)
    plt.show()