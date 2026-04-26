# src/noise.py
import numpy as np
import pandas as pd
from sklearn.neighbors import KDTree
from sklearn.svm import SVC

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

def nonlinearwise_sample(df, frac, seed) -> pd.DataFrame:
    X = df.iloc[:, :-1].values
    y = df[["class"]].values.ravel()
    svm = SVC(kernel="rbf", random_state=seed)
    svm.fit(X, y)
    df_dists = pd.DataFrame(svm.decision_function(X), columns=["distance"]
        ).abs().sort_values(by="distance")
    sample_indices = df_dists.head(int(df.shape[0] * frac)).index
    return df.loc[sample_indices]

def get_sample(df, frac:float, sample_type, seed=0) -> pd.DataFrame:
    match sample_type:
        case "random":
            return df.sample(frac=frac, random_state=seed)
        case "neighborwise":
            return neighborwise_sample(df, frac)
        case "nonlinearwise":
            return nonlinearwise_sample(df, frac, seed)
