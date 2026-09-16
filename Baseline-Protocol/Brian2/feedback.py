"""
Rules shared by the Brian2 and Hardware versions.

This file is identical in both folders. If you change a rule here,
change it in the other folder too so the two versions stay comparable.
"""
import numpy as np

N_CHANNELS = 60
LEFT_CHANNELS = list(range(0, 30))    # first half of the MEA
RIGHT_CHANNELS = list(range(30, 60))  # second half of the MEA

# CartPole ends the episode at 0.2095 rad (12 degrees). Inside this limit the
# pole counts as "balanced" and the neurons get predictable stimulation;
# outside it they get unpredictable stimulation.
BALANCE_LIMIT_RAD = 0.15


def is_balanced(pole_angle):
    return abs(pole_angle) < BALANCE_LIMIT_RAD


def lean_side(pole_angle):
    """Which side of the MEA to stimulate. The pole leans left when angle < 0."""
    return "left" if pole_angle < 0 else "right"


def choose_action(spike_counts):
    """
    Decode an action from spike counts (one number per channel).
    More spikes on the left half  -> action 0 (push cart left).
    More spikes on the right half -> action 1 (push cart right).
    A tie is broken randomly.
    """
    left = spike_counts[LEFT_CHANNELS].sum()
    right = spike_counts[RIGHT_CHANNELS].sum()
    if left > right:
        return 0
    if right > left:
        return 1
    return np.random.randint(2)
