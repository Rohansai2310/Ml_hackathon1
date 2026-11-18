"""The five comparable baselines and bounded CPU tuning spaces."""

from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

MODELS = ["logistic_regression", "gaussian_nb", "decision_tree", "random_forest", "adaboost"]


def build_model(name, seed=42, jobs=1, params=None):
    params = params or {}
    if name == "logistic_regression":
        estimator = LogisticRegression(max_iter=2000, random_state=seed, **params)
    elif name == "gaussian_nb":
        estimator = GaussianNB(**params)
    elif name == "decision_tree":
        estimator = DecisionTreeClassifier(random_state=seed, **params)
    elif name == "random_forest":
        estimator = RandomForestClassifier(random_state=seed, n_jobs=jobs, **({"n_estimators": 200} | params))
    elif name == "adaboost":
        settings = {"n_estimators": 100, "learning_rate": 0.5} | params
        depth = settings.pop("tree_depth", 1)
        estimator = AdaBoostClassifier(
            estimator=DecisionTreeClassifier(max_depth=depth, random_state=seed),
            random_state=seed, **settings,
        )
    else:
        raise ValueError(f"Unknown model: {name}")
    steps = [("imputer", SimpleImputer(strategy="median", keep_empty_features=True))]
    if name in ["logistic_regression", "gaussian_nb"]:
        steps.append(("scaler", StandardScaler()))
    return Pipeline(steps + [("classifier", estimator)])


def suggest_params(trial, name):
    if name == "logistic_regression":
        return {"C": trial.suggest_float("C", 1e-3, 100, log=True)}
    if name == "gaussian_nb":
        return {"var_smoothing": trial.suggest_float("var_smoothing", 1e-12, 1e-3, log=True)}
    if name == "decision_tree":
        return {
            "max_depth": trial.suggest_int("max_depth", 2, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 30),
        }
    if name == "random_forest":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 400, step=50),
            "max_depth": trial.suggest_int("max_depth", 3, 24),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", 0.5, 1.0]),
        }
    return {
        "tree_depth": trial.suggest_int("tree_depth", 1, 3),
        "n_estimators": trial.suggest_int("n_estimators", 50, 300, step=50),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 1.0, log=True),
    }
