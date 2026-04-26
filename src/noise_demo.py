# src/data_analysis.py
import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.neighbors import KDTree

def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        "noise demo",
        description="""Generates a diagram of the different noise algorithms applyed in
the experiment.""")
    parser.add_argument("-frac", type=float, default=0.10)
    return parser

def interclass_nearest_neighbor(df) -> np.ndarray:
    """Returns a `numpy.ndarray` of euclidean distances of every point to the
    nearest neighbor from the *other* class."""
    df0 = df[df["class"] == 0][["X", "Y"]]
    df1 = df[df["class"] == 1][["X", "Y"]]
    tree0 = KDTree(df0.values, leaf_size=2, metric="euclidean")
    tree1 = KDTree(df1.values, leaf_size=2, metric="euclidean")

    nn_dist = np.ndarray((df.shape[0], 1))
    nn_dist[df0.index] = tree1.query(df0.values, k=1)[0]
    nn_dist[df1.index] = tree0.query(df1.values, k=1)[0]
    return nn_dist

def intraclass_nearest_neighbor(df) -> np.ndarray:
    """Returns a `numpy.ndarray` of euclidean distances of every point to the
    nearest neighbor from the *same* class."""
    nn_dist = np.ndarray((df.shape[0], 1))
    for name, group in df.groupby("class"):
        tree = KDTree(group.values, leaf_size=2, metric="euclidean")
        # Setting k=2 because k=1 is the given point itself.
        dist = tree.query(group.values, k=2)[0]
        nn_dist[group.index] = dist[:, 1].reshape((len(dist), 1))
    return nn_dist

def neighborwise_sample(df, frac) -> pd.DataFrame:
    nn_dists = intraclass_nearest_neighbor(df) / interclass_nearest_neighbor(df)
    df_dists = pd.DataFrame(nn_dists, columns=["distance"]
        ).sort_values(by="distance", ascending=False)
    sample_indices = df_dists.head(int(df.shape[0] * frac)).index
    return df.loc[sample_indices]

def get_sample(df, frac:float, sample_type, seed=0) -> pd.DataFrame:
    match sample_type:
        case "random":
            return df.sample(frac=frac, random_state=seed)
        case "neighborwise":
            return neighborwise_sample(df, frac)

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
    label_noise_sample = get_sample(df, frac, "random")
    label_noise_sample["class"] = 1 - label_noise_sample["class"]
    df_noisy = df.copy()
    df_noisy.loc[label_noise_sample.index, :] = label_noise_sample
    sns.scatterplot(data=df_noisy, x="X", y="Y", hue="class", s=75, ax=axes[0, 1])
    sns.scatterplot(data=label_noise_sample, x="X", y="Y", s=150, facecolors="none",
                    edgecolor='r', linestyle='--', linewidth=1, ax=axes[0, 1])

    axes[1, 0].set_title(f"Neighborwise label noise ({int(frac*100)}%)")
    label_noise_sample = get_sample(df, frac, "neighborwise")
    label_noise_sample["class"] = 1 - label_noise_sample["class"]
    df_noisy = df.copy()
    df_noisy.loc[label_noise_sample.index, :] = label_noise_sample
    sns.scatterplot(data=df_noisy, x="X", y="Y", hue="class", s=75, ax=axes[1, 0])
    sns.scatterplot(data=label_noise_sample, x="X", y="Y", s=150, facecolors="none",
                    edgecolor='r', linestyle='--', linewidth=1, ax=axes[1, 0])

    for ax in axes.flat:
        ax.legend()
        ax.get_legend().remove()

    plt.show()
