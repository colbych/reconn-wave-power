"""
Plotting helpers for PSDs and spectrograms.
"""
import matplotlib.pyplot as plt
import numpy as np

def plot_psd(f, Pxx, label=None, ax=None, scale="log"):
    if ax is None:
        fig, ax = plt.subplots()
    ax.plot(f, Pxx, label=label)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("Power")
    if label:
        ax.legend()
    return ax