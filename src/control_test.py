# src/control_test.py
# Used for training/evaluating models on unaltered data
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from models import BaseLearner, AdaBoost, GradientBoosting, RandomForest

if __name__ == "__main__":
    data_path = "data/clean-telco-churn.csv"
    try:
        data = pd.read_csv(data_path, delimiter=",")
    except FileNotFoundError:
        print(f"ERROR: Cleaned dataset '{data_path}' not found. Did you run module clean_data?")

    """TODO
    Add Feature Selection Here (?)
    """

    learner = BaseLearner()
    adaboost = AdaBoost(learner)
    gradient_boosting = GradientBoosting()
    random_forest = RandomForest()

    """TODO
    Add Training Process
    """
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1:]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=15)

    learner.model.fit(X_train, y_train)
    y_pred = learner.model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy = {accuracy:.2f}")

    """TODO
    Add Evaluation Process
    """

    """TODO
    Save Results to a File
    """
