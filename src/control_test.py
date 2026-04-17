# src/control_test.py
# Used for training/evaluating models on unaltered data
import pandas as pd
from sklearn.model_selection import train_test_split

from models import BaseLearner, AdaBoost, GradientBoosting, RandomForest

if __name__ == "__main__":
    data_path = "data/clean-telco-churn.csv"
    try:
        data = pd.read_csv(data_path, delimiter=",")
    except FileNotFoundError:
        print(f"ERROR: Cleaned dataset '{data_path}' not found. Did you run module clean_data?")

    learner = BaseLearner()
    adaboost = AdaBoost(learner)
    gradient_boosting = GradientBoosting()
    random_forest = RandomForest()

    X = data.iloc[:, :-1]
    y = data.iloc[:, -1:]

    training_scores = []
    testing_scores = []
    for seed in range(0, 100):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)

        learner.model.fit(X_train, y_train)
        training_scores.append(learner.model.score(X_train, y_train))
        testing_scores.append(learner.model.score(X_test, y_test))
        # print(f"Seed {seed}: Training Accuracy = {training_scores[-1]*100:2.1f}%")
        # print(f"Seed {seed}: Testing Accuracy = {testing_scores[-1]*100:2.1f}%")
    avg_training_score = sum(training_scores)/len(training_scores)
    avg_testing_score = sum(testing_scores)/len(testing_scores)
    print(f"AVG Training Accuracy: {avg_training_score*100:2.1f}%")
    print(f"AVG Testing Accuracy: {avg_testing_score*100:2.1f}%")

    """TODO
    Add Evaluation Process
    """

    """TODO
    Save Results to a File
    """
