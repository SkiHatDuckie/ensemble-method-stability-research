# src/models/base_learner.py
# Model used as the base learner for all ensemble methods.
from sklearn import tree
from sklearn.ensemble import AdaBoostClassifier, \
    HistGradientBoostingClassifier, RandomForestClassifier


class BaseLearner:
    def __init__(self) -> None:
        self.model = tree.DecisionTreeClassifier(max_depth=1)


class AdaBoost:
    def __init__(self, model) -> None:
        self.method = AdaBoostClassifier(estimator=model, n_estimators=100)


class GradientBoosting:
    def __init__(self) -> None:
        self.method = HistGradientBoostingClassifier()


class RandomForest:
    def __init__(self) -> None:
        self.method = RandomForestClassifier(n_estimators=100)
