# src/control_test.py
# Used for training/evaluating models on unaltered data
import pandas as pd

if __name__ == "__main__":
    data_path = "data/WA_Fn-UseC_-Telco-Customer-Churn.csv"
    data = pd.read_csv(data_path, delimiter=",")
