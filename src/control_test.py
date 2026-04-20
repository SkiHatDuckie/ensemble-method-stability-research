# src/control_test.py
# Used for training/evaluating models on unaltered data
import os
from pathlib import Path
import time

from sklearn.model_selection import train_test_split
from ucimlrepo import fetch_ucirepo

from models import BaseLearner, AdaBoost, GradientBoosting, RandomForest

def create_results_filepath(location, prefix="results") -> Path:
    filename = f"{prefix}_{time.localtime()}.res"
    filepath = location+filename
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    return filepath

def avg(list_) -> float:
    """Returns the mean from a list of numbers `list_`."""
    return sum(list_) / len(list_)

def train_test_loop(model, num_runs) -> tuple[list, list]:
    training_scores = []
    testing_scores = []
    for seed in range(0, num_runs):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                            random_state=seed)
        model.fit(X_train, y_train)
        training_scores.append(model.score(X_train, y_train))
        testing_scores.append(model.score(X_test, y_test))
    return training_scores, testing_scores

if __name__ == "__main__":
    banknote_authentication = fetch_ucirepo(id=267)
    X = banknote_authentication.data.features
    y = banknote_authentication.data.targets.values.ravel()

    results_path = create_results_filepath("results/", prefix="control")

    learner = BaseLearner()
    adaboost = AdaBoost(learner.model)
    gradient_boosting = GradientBoosting()
    random_forest = RandomForest()
    # methods = (learner, adaboost, gradient_boosting, random_forest)
    methods = (learner,)
    runs = 100
    for method in methods:
        print(f"Method: {method}")
        time_start = time.perf_counter()
        training_scores, testing_scores = train_test_loop(method.model, runs)
        print(f"runs: {runs}\t time: {time.perf_counter() - time_start:.3f} sec")
        print(f"avg. training accuracy: {avg(training_scores)*100:.2f}%")
        print(f"avg. testing accuracy: {avg(testing_scores)*100:.2f}%\n")

    """TODO
    Save Results to a File
    """
