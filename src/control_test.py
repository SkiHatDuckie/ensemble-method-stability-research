# src/control_test.py
# Used for training/evaluating models on unaltered data
import argparse
import os
from pathlib import Path
import sys
import time
import typing

from sklearn.model_selection import train_test_split
from ucimlrepo import fetch_ucirepo

from metrics import Metric, MetricActions
from models import BaseLearner, AdaBoost, GradientBoosting, RandomForest

def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        "control_test",
        description="""Run the main train-test loop. Records final results in directory
`results/`""")
    parser.add_argument("-noise", type=float, default=0.0)
    parser.add_argument("-debug", action="store_true")
    return parser

def create_results_filepath(location, prefix="results") -> Path:
    filename = f"{prefix}_{time.strftime('%Y%m%d-%H%M%S')}.txt"
    filepath = location+filename
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    return filepath

def set_output_stream(debug, noise) -> typing.TextIO:
    if debug:
        return sys.stdout
    prefix = "control" if noise <= 0.0 else f"noise{int(noise * 100)}"
    return open(create_results_filepath("results/", prefix=prefix),
                "a",
                encoding="utf-8")

def write_results(stream, method, num_runs, *metrics_) -> None:
    stream.write(f"Method: {method}\n")
    stream.write(f"runs: {num_runs}\n")
    for metric in metrics_:
        stream.write(f"{metric}")
    stream.write("\n")

def train_test_loop(method, num_runs, out_stream, noise=0.0) -> None:
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

    write_results(out_stream, method, num_runs, *results_metrics)

if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    out_stream = set_output_stream(args.debug, args.noise)

    banknote_authentication = fetch_ucirepo(id=267)
    X = banknote_authentication.data.features
    X = X.drop("entropy", axis=1)
    y = banknote_authentication.data.targets.values.ravel()

    learner = BaseLearner()
    adaboost = AdaBoost(learner.model)
    gradient_boosting = GradientBoosting()
    random_forest = RandomForest()
    methods = (learner, adaboost, gradient_boosting, random_forest)
    runs = 100
    for method in methods:
        print(f"Running method: {method}")
        train_test_loop(method, runs, out_stream, args.noise)
