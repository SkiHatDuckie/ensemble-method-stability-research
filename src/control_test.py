# src/control_test.py
# Used for training/evaluating models on unaltered data
import os
from pathlib import Path
import time

from sklearn.model_selection import train_test_split
from ucimlrepo import fetch_ucirepo

from metrics import Metric, MetricActions
from models import BaseLearner, AdaBoost, GradientBoosting, RandomForest

def create_results_filepath(location, prefix="results") -> Path:
    filename = f"{prefix}_{time.strftime('%Y%m%d-%H%M%S')}.res"
    filepath = location+filename
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    return filepath

def write_results(filepath, method, num_runs, *metrics_):
    with open(filepath, "a", encoding="utf-8") as f:
        f.write(f"Method: {method}\n")
        f.write(f"runs: {num_runs}\n")
        for metric in metrics_:
            f.write(f"{metric}")
        f.write("\n")

def train_test_loop(method, num_runs, results_path) -> None:
    training_scores = Metric(name="training accuracy",
                             actions=[MetricActions.PERCENT_AVERAGE],
                             decimal_precision=2)
    testing_scores = Metric(name="testing accuracy",
                            actions=[MetricActions.PERCENT_AVERAGE],
                            decimal_precision=2)
    training_times = Metric(name="training time",
                            actions=[MetricActions.AVERAGE, MetricActions.TOTAL],
                            suffix="sec")
    results_metrics = [training_scores, testing_scores, training_times]

    for seed in range(0, num_runs):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                            random_state=seed)
        time_start = time.perf_counter()
        method.model.fit(X_train, y_train)

        training_times.data.append(time.perf_counter() - time_start)
        training_scores.data.append(method.model.score(X_train, y_train))
        testing_scores.data.append(method.model.score(X_test, y_test))

    write_results(results_path, method, num_runs, *results_metrics)

if __name__ == "__main__":
    banknote_authentication = fetch_ucirepo(id=267)
    X = banknote_authentication.data.features
    y = banknote_authentication.data.targets.values.ravel()
    results_path = create_results_filepath("results/", prefix="control")

    learner = BaseLearner()
    adaboost = AdaBoost(learner.model)
    gradient_boosting = GradientBoosting()
    random_forest = RandomForest()
    methods = (learner, adaboost, gradient_boosting, random_forest)
    runs = 100
    for method in methods:
        print(f"Running method: {method}")
        train_test_loop(method, runs, results_path)
