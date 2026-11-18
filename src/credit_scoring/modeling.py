import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

HISTORY_PREFIXES = ("BUREAU_", "POS_", "INST_", "CC_", "PREV_", "HAS_")


def feature_groups(frame):
    application = [
        column for column in frame if not column.startswith(HISTORY_PREFIXES)
    ]
    bureau = [
        column
        for column in frame
        if column.startswith("BUREAU_") or column == "HAS_BUREAU_HISTORY"
    ]
    return {
        "application": application,
        "application_bureau": application + bureau,
        "all_history": frame.columns.tolist(),
    }


def build_logistic(c_value=1.0):
    numeric = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())
    categorical = make_pipeline(
        SimpleImputer(strategy="constant", fill_value="Missing"),
        OneHotEncoder(handle_unknown="ignore", sparse_output=True),
    )
    preprocessing = ColumnTransformer(
        [
            ("numeric", numeric, make_column_selector(dtype_include=np.number)),
            ("categorical", categorical, make_column_selector(dtype_include="object")),
        ],
        sparse_threshold=1.0,
    )
    return make_pipeline(
        preprocessing,
        LogisticRegression(C=c_value, max_iter=2000, random_state=42, solver="lbfgs"),
    )


def prepare_catboost(frame):
    result = frame.copy()
    categorical = result.select_dtypes("object").columns.tolist()
    result[categorical] = result[categorical].fillna("Missing").astype(str)
    return result, categorical


def build_catboost(categorical):
    return CatBoostClassifier(
        iterations=500,
        depth=6,
        learning_rate=0.05,
        loss_function="Logloss",
        random_seed=42,
        thread_count=4,
        verbose=False,
        allow_writing_files=False,
        cat_features=categorical,
    )


def probability_metrics(target, probabilities):
    return {
        "roc_auc": roc_auc_score(target, probabilities),
        "average_precision": average_precision_score(target, probabilities),
        "brier": brier_score_loss(target, probabilities),
    }


def probability_logit(probabilities):
    probabilities = np.clip(probabilities, 1e-6, 1 - 1e-6)
    return np.log(probabilities / (1 - probabilities)).reshape(-1, 1)


def risk_deciles(target, probabilities):
    table = pd.DataFrame({"target": np.asarray(target), "probability": probabilities})
    table["decile"] = (
        pd.qcut(table["probability"], 10, labels=False, duplicates="drop") + 1
    )
    return (
        table.groupby("decile", observed=True)
        .agg(
            clients=("target", "size"),
            default_rate=("target", "mean"),
            mean_probability=("probability", "mean"),
            defaults=("target", "sum"),
        )
        .reset_index()
    )
