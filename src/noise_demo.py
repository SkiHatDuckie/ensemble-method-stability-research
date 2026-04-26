# src/data_analysis.py
import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import noise

def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        "noise demo",
        description="""Generates a diagram of the different noise algorithms applied in
the experiment.""")
    parser.add_argument("-noise", type=float, default=0.1)
    return parser

def plot_noisy_data(df, noisy_sample, ax):
    noisy_sample["class"] = 1 - noisy_sample["class"]
    df.loc[noisy_sample.index, :] = noisy_sample
    sns.scatterplot(data=df, x="X", y="Y", hue="class", s=75, ax=ax)
    sns.scatterplot(data=noisy_sample, x="X", y="Y", s=150, facecolors="none",
                    edgecolor='r', linestyle='--', linewidth=1, ax=ax)

if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    frac = args.frac

    np.random.seed(5)
    df = pd.DataFrame(np.random.rand(100, 2), columns=["X", "Y"])
    df["class"] = (df["Y"] >= df["X"]).astype(int)

    fig, axes = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(10, 8))
    plt.setp(axes, xlim=(0, 1), ylim=(0, 1), xticks=[], yticks=[])
    for ax in axes.flat:
        ax.axline((0, 0), slope=1, color="gray", linestyle="--")

    axes[0, 0].set_title("Original")
    sns.scatterplot(data=df, x="X", y="Y", hue="class", s=75, ax=axes[0, 0])

    axes[0, 1].set_title(f"Random label noise ({int(frac*100)}%)")
    label_noise_sample = noise.get_sample(df, frac, "random")
    plot_noisy_data(df.copy(), label_noise_sample, axes[0, 1])

    axes[1, 0].set_title(f"Neighborwise label noise ({int(frac*100)}%)")
    label_noise_sample = noise.get_sample(df, frac, "neighborwise")
    plot_noisy_data(df.copy(), label_noise_sample, axes[1, 0])

    axes[1, 1].set_title(f"Non Linearwise label noise ({int(frac*100)}%)")
    label_noise_sample = noise.get_sample(df, frac, "nonlinearwise")
    plot_noisy_data(df.copy(), label_noise_sample, axes[1, 1])

    for ax in axes.flat:
        ax.legend()
        ax.get_legend().remove()

    plt.show()
