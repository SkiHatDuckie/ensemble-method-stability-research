# src/control_test.py
# Used for training/evaluating models on unaltered data
import argparse
import os
from pathlib import Path
import sys
import time
import typing

import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from ucimlrepo import fetch_ucirepo

from metrics import Metric, MetricActions
from models import BaseLearner, AdaBoost, GradientBoosting, RandomForest
from noise import get_sample

def get_probability_scores(model, X_test):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X_test)[:, 1]
    if hasattr(model, "decision_function"):
        return model.decision_function(X_test)
    return model.predict(X_test)

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

def create_csv_results_filepath(location, prefix="control_raw") -> Path:
    filename = f"{prefix}_{time.strftime('%Y%m%d-%H%M%S')}.csv"
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

def write_results(stream, sample_type, *metrics_) -> None:
    stream.write(f"Sample type: {sample_type}\n")
    for metric in metrics_:
        stream.write(f"{metric}")
    stream.write("\n")

def inject_label_noise(X, y, noise, sample_type) -> pd.Series:
    noisy_sample = get_sample(pd.concat([X, y], axis=1), noise, sample_type)
    noisy_sample["class"] = 1 - noisy_sample["class"]
    y.loc[noisy_sample.index, :] = noisy_sample["class"]
    return y

def train_test_loop(X, y, method, num_runs, out_stream, noise, raw_records) -> None:
    training_scores = Metric(name="training accuracy",
                             actions=[MetricActions.PERCENT_AVERAGE],
                             decimal_precision=2)
    testing_scores = Metric(name="testing accuracy",
                            actions=[MetricActions.PERCENT_AVERAGE],
                            decimal_precision=2)
    testing_f1_scores = Metric(name="testing F1 score",
                               actions=[MetricActions.AVERAGE],
                               decimal_precision=3)
    testing_roc_auc = Metric(name="testing ROC-AUC",
                             actions=[MetricActions.AVERAGE],
                             decimal_precision=3)
    training_times = Metric(name="training time",
                            actions=[MetricActions.AVERAGE, MetricActions.TOTAL],
                            suffix="sec")
    results_metrics = [training_scores, testing_scores, testing_f1_scores, testing_roc_auc, training_times]

    sample_types = ["control (none)"]
    if noise > 0.0:
        sample_types = ["random", "neighborwise", "nonlinearwise"]
    for sample_type in sample_types:
        for seed in range(0, num_runs):
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                                random_state=seed)
            if noise > 0.0:
                X_train = X_train.reset_index(drop=True)
                y_train = y_train.reset_index(drop=True)
                y_train = inject_label_noise(X_train, y_train, noise, sample_type)

            time_start = time.perf_counter()
            method.model.fit(X_train, y_train.values.ravel())
            training_time = time.perf_counter() - time_start

            y_pred = method.model.predict(X_test)
            y_score = get_probability_scores(method.model, X_test)
            training_accuracy = method.model.score(X_train, y_train)
            testing_accuracy = method.model.score(X_test, y_test)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            roc_auc = roc_auc_score(y_test, y_score)

            training_times.data.append(training_time)
            training_scores.data.append(training_accuracy)
            testing_scores.data.append(testing_accuracy)
            testing_f1_scores.data.append(f1)
            testing_roc_auc.data.append(roc_auc)

            raw_records.append({
                "method": str(method),
                "sample_type": sample_type,
                "seed": seed,
                "training_accuracy": training_accuracy,
                "testing_accuracy": testing_accuracy,
                "training_time": training_time,
                "f1_score": f1,
                "roc_auc": roc_auc,
            })

        write_results(out_stream, sample_type, *results_metrics)

if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    out_stream = set_output_stream(args.debug, args.noise)
    results_header = f"=== Testing with {int(args.noise * 100)}% noise ===\n"
    print(results_header)
    out_stream.write(results_header + "\n")

    banknote_authentication = fetch_ucirepo(id=267)
    X = banknote_authentication.data.features
    X = X.drop("entropy", axis=1)
    y = banknote_authentication.data.targets

    learner = BaseLearner()
    decision_tree = BaseLearner(depth=5)
    adaboost = AdaBoost(learner.model)
    gradient_boosting = GradientBoosting()
    random_forest = RandomForest()
    methods = (decision_tree, adaboost, gradient_boosting, random_forest)
    runs = 100

    raw_records = []
    for method in methods:
        print(f"Running method: {method}")
        out_stream.write(f"Method: {method}\n")
        out_stream.write(f"Runs per sample type: {runs}\n")
        train_test_loop(X, y, method, runs, out_stream, args.noise, raw_records)

    raw_results_path = create_csv_results_filepath("results/", prefix="control_raw")
    pd.DataFrame(raw_records).to_csv(raw_results_path, index=False)
    print(f"Saved raw control results to {raw_results_path}")
