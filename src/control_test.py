# src/control_test.py
# Used for training/evaluating models on unaltered data
import pandas as pd

if __name__ == "__main__":
    data_path = "data/clean-telco-churn.csv"
    try:
        data = pd.read_csv(data_path, delimiter=",")
    except FileNotFoundError:
        print(f"ERROR: Cleaned dataset '{data_path}' not found. Did you run module clean_data?")

    """TODO
    Add Feature Selection Here
    """

    """TODO
    Add Training Process
    """

    """TODO
    Add Evaluation Process
    """

    """TODO
    Save Results to a File
    """
