# src/models/base_learner.py
from sklearn import tree
from sklearn.ensemble import AdaBoostClassifier, \
    HistGradientBoostingClassifier, RandomForestClassifier


class BaseLearner:
    def __init__(self) -> None:
        self.model = tree.DecisionTreeClassifier(max_depth=2)

    def __str__(self) -> str:
        return "DecisionTreeClassifier"


class AdaBoost:
    def __init__(self, model) -> None:
        self.model = AdaBoostClassifier(estimator=model, n_estimators=30)
    
    def __str__(self) -> str:
        return "AdaBoostClassifier"


class GradientBoosting:
    def __init__(self) -> None:
        self.model = HistGradientBoostingClassifier(learning_rate=0.2, max_depth=2)
    
    def __str__(self) -> str:
        return "HistGradientBoostingClassifier"


class RandomForest:
    def __init__(self) -> None:
        self.model = RandomForestClassifier(n_estimators=30, max_depth=2, random_state=0)
    
    def __str__(self) -> str:
        return "RandomForestClassifier"
