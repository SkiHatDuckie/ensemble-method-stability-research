# src/control_test.py
# Used for training/evaluating models on unaltered data
from sklearn.model_selection import train_test_split
from ucimlrepo import fetch_ucirepo

from models import BaseLearner, AdaBoost, GradientBoosting, RandomForest

if __name__ == "__main__":
    banknote_authentication = fetch_ucirepo(id=267)
    # try:
    #     data = pd.read_csv(data_path, delimiter=",")
    # except FileNotFoundError:
    #     print(f"ERROR: Cleaned dataset '{data_path}' not found. Did you run module clean_data?")

    learner = BaseLearner()
    adaboost = AdaBoost(learner.model)
    gradient_boosting = GradientBoosting()
    random_forest = RandomForest()

    X = banknote_authentication.data.features
    y = banknote_authentication.data.targets
    y = y.values.ravel()

    methods = (learner, adaboost, gradient_boosting, random_forest)
    for method in methods:
        training_scores = []
        testing_scores = []
        print(f"Method/Model: {method}")
        for seed in range(0, 100):
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)
            

            method.model.fit(X_train, y_train)
            training_scores.append(method.model.score(X_train, y_train))
            testing_scores.append(method.model.score(X_test, y_test))
            # print(f"Seed {seed}: Training Accuracy = {training_scores[-1]*100:2.1f}%")
            # print(f"Seed {seed}: Testing Accuracy = {testing_scores[-1]*100:2.1f}%")
        avg_training_score = sum(training_scores)/len(training_scores)
        avg_testing_score = sum(testing_scores)/len(testing_scores)
        print(f"AVG Training Accuracy: {avg_training_score*100:2.1f}%")
        print(f"AVG Testing Accuracy: {avg_testing_score*100:2.1f}%")

    """TODO
    Save Results to a File
    """
