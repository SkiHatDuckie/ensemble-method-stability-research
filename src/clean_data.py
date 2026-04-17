# src/clean_data.py
# Cleaning and preprocessing the raw CSV for testing.
# Run this before running any tests.
import json

import pandas as pd

if __name__ == "__main__":
    data = pd.read_csv("data/WA_Fn-UseC_-Telco-Customer-Churn.csv", delimiter=",")

    """TODO
    Do Preprocessing
    """

    # Remap all non-numeric columns to numeric using lookup tables
    with open("src/category_remap.json", "r", encoding="utf-8") as f:
        tables = json.load(f)
        for col in tables.keys():
            data[col] = data[col].map(tables[col])

    with open("data/clean-telco-churn.csv", "w", encoding="utf-8") as f:
        data.to_csv(f)
