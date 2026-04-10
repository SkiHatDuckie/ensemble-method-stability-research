# src/clean_data.py
# Cleaning and preprocessing the raw CSV for testing.
# Run this before running any tests.

import pandas as pd

if __name__ == "__main__":
    data_path = "data/WA_Fn-UseC_-Telco-Customer-Churn.csv"
    data = pd.read_csv(data_path, delimiter=",")

    """TODO
    Do Preprocessing
    """

    clean_data_path = "data/clean-telco-churn.csv"
    with open(clean_data_path, "w", encoding="utf-8") as f:
        data.to_csv(f)
