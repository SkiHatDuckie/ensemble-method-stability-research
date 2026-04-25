# src/data_analysis.py
import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

def get_sample(df, frac:float, sample_type, seed=0) -> pd.DataFrame:
    match sample_type:
        case "random":
            return df.sample(frac=frac, random_state=seed)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "noise demo",
        description="""Generates a diagram of the different noise algorithms applyed in
the experiment.""")
    parser.add_argument("-frac", type=float, default=0.10)
    args = parser.parse_args()
    frac = args.frac

    np.random.seed(3)
    df = pd.DataFrame(np.random.rand(100, 2), columns=["X", "Y"])
    df["class"] = (df["Y"] >= df["X"]).astype(int)

    fig, axes = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(10, 4))
    plt.setp(axes, xlim=(0, 1), ylim=(0, 1), xticks=[], yticks=[])
    for ax in axes.flat:
        ax.axline((0, 0), slope=1, color="gray", linestyle="--")

    axes[0].set_title("Original")
    sns.scatterplot(data=df, x="X", y="Y", hue="class", s=75, ax=axes[0])

    axes[1].set_title("Random Label Noise")
    label_noise_sample = get_sample(df, frac, "random")
    label_noise_sample["class"] = 1 - label_noise_sample["class"]
    df.loc[label_noise_sample.index, :] = label_noise_sample

    sns.scatterplot(data=df, x="X", y="Y", hue="class", s=75, ax=axes[1])
    sns.scatterplot(data=label_noise_sample, x="X", y="Y", s=150, facecolors="none",
                    edgecolor='r', linestyle='--', linewidth=1, ax=axes[1])

    for ax in axes.flat:
        ax.legend()
        ax.get_legend().remove()

    plt.show()
