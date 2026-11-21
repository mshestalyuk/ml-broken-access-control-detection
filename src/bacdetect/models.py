from scipy.stats import loguniform, randint
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler
from sklearn.svm import SVC

from bacdetect.config import RANDOM_STATE
from bacdetect.features import BacIdorFeatures
from bacdetect.preprocessing import code_tokenizer

MODEL_NAMES = ("lr", "rf", "svm")


def build_features(max_features: int = 50_000) -> FeatureUnion:
    tfidf = TfidfVectorizer(
        tokenizer=code_tokenizer,
        token_pattern=None,
        lowercase=False,  # tokenizer already lowercases identifiers
        ngram_range=(1, 3),
        max_features=max_features,
        min_df=1,
        sublinear_tf=True,
    )
    handcrafted = Pipeline([("extract", BacIdorFeatures()), ("scale", MinMaxScaler())])
    return FeatureUnion([("tfidf", tfidf), ("bac_idor", handcrafted)])


def build_classifier(name: str):
    if name == "lr":
        return LogisticRegression(
            solver="liblinear",
            class_weight="balanced",
            max_iter=2000,
            random_state=RANDOM_STATE,
        )
    if name == "rf":
        return RandomForestClassifier(
            n_estimators=300, class_weight="balanced", n_jobs=-1, random_state=RANDOM_STATE
        )
    if name == "svm":
        # probability=False: probabilities come from CalibratedClassifierCV in train.py
        return SVC(kernel="linear", class_weight="balanced", random_state=RANDOM_STATE)
    raise ValueError(f"unknown model {name!r}, expected one of {MODEL_NAMES}")


def build_pipeline(name: str, max_features: int = 50_000) -> Pipeline:
    steps = [("features", build_features(max_features))]
    if name == "svm":
        steps.append(("scale", MaxAbsScaler()))  # keeps sparsity, helps SVM convergence
    steps.append(("clf", build_classifier(name)))
    return Pipeline(steps)


def param_distributions(name: str) -> dict:
    """Search spaces from the 'Algorytmy' table, for RandomizedSearchCV."""
    common = {"features__tfidf__ngram_range": [(1, 1), (1, 2), (1, 3)]}
    if name == "lr":
        return common | {
            "clf__C": loguniform(1e-3, 1e2),
            "clf__l1_ratio": [0.0, 1.0],  # 0 = L2, 1 = L1 (sklearn >= 1.8 API)
            "clf__class_weight": ["balanced", None],
        }
    if name == "rf":
        return common | {
            "clf__n_estimators": randint(100, 600),
            "clf__max_depth": [None, 10, 20, 40],
            "clf__min_samples_leaf": randint(1, 10),
            "clf__class_weight": ["balanced", "balanced_subsample", None],
        }
    if name == "svm":
        return common | {
            "clf__kernel": ["linear", "rbf"],
            "clf__C": loguniform(1e-2, 1e2),
            "clf__gamma": ["scale", "auto"],
        }
    raise ValueError(name)
