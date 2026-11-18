"""Fixed 60/20/20 split, three CV folds, feature ablation and calibration."""

import json
import platform
import shutil
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, train_test_split

from credit_scoring.modeling import (
    build_catboost,
    build_logistic,
    feature_groups,
    prepare_catboost,
    probability_logit,
    probability_metrics,
    risk_deciles,
)


def main():
    report = Path("reports/evaluation")
    models = Path("models/trained")
    report.mkdir(parents=True, exist_ok=True)
    models.mkdir(parents=True, exist_ok=True)
    for filename in [
        "source_quality.csv",
        "feature_missingness.csv",
        "data_quality.json",
    ]:
        shutil.copyfile(Path("processed") / filename, report / filename)
    data = pd.read_parquet("processed/train_features.parquet")
    target = data["TARGET"]
    features = data.drop(columns=["TARGET", "SK_ID_CURR"])
    development, holdout = train_test_split(
        np.arange(len(data)), test_size=0.2, stratify=target, random_state=42
    )
    train, calibration = train_test_split(
        development, test_size=0.25, stratify=target.iloc[development], random_state=42
    )
    split = pd.Series("train", index=data.index)
    split.iloc[calibration] = "calibration"
    split.iloc[holdout] = "holdout"
    data[["SK_ID_CURR", "TARGET"]].assign(split=split).to_csv(
        report / "split_membership.csv", index=False
    )
    groups = feature_groups(features)
    (report / "feature_groups.json").write_text(json.dumps(groups, indent=2))
    cat_features, categorical = prepare_catboost(features)
    folds = list(
        StratifiedKFold(3, shuffle=True, random_state=42).split(
            features.iloc[train], target.iloc[train]
        )
    )
    rows = []
    specifications = [("logistic", "all_history", c) for c in [0.1, 1.0]]
    specifications += [("catboost", group, None) for group in groups]
    for family, group, c_value in specifications:
        columns = groups[group]
        model_data = features if family == "logistic" else cat_features
        for fold, (fit_positions, valid_positions) in enumerate(folds, 1):
            fit, valid = train[fit_positions], train[valid_positions]
            model = (
                build_logistic(c_value)
                if family == "logistic"
                else build_catboost([name for name in categorical if name in columns])
            )
            started = time.perf_counter()
            model.fit(model_data.iloc[fit][columns], target.iloc[fit])
            elapsed = time.perf_counter() - started
            probabilities = model.predict_proba(model_data.iloc[valid][columns])[:, 1]
            row = {
                "model": family,
                "group": group,
                "C": c_value,
                "fold": fold,
                "features": len(columns),
                "fit_seconds": elapsed,
                "iterations": int(model[-1].n_iter_[0])
                if family == "logistic"
                else 500,
                **probability_metrics(target.iloc[valid], probabilities),
            }
            rows.append(row)
            pd.DataFrame(rows).to_csv(report / "cv_results.csv", index=False)
            print(row, flush=True)

    cv = pd.DataFrame(rows)
    best_c = cv.query("model == 'logistic'").groupby("C")["roc_auc"].mean().idxmax()
    holdout_scores, predictions = [], {}
    for family, group in [("logistic", "all_history")] + [
        ("catboost", g) for g in groups
    ]:
        columns = groups[group]
        model_data = features if family == "logistic" else cat_features
        model = (
            build_logistic(best_c)
            if family == "logistic"
            else build_catboost([name for name in categorical if name in columns])
        )
        started = time.perf_counter()
        model.fit(model_data.iloc[train][columns], target.iloc[train])
        probabilities = model.predict_proba(model_data.iloc[holdout][columns])[:, 1]
        name = f"{family}_{group}"
        predictions[name] = probabilities
        holdout_scores.append(
            {
                "model": name,
                "features": len(columns),
                "fit_seconds": time.perf_counter() - started,
                **probability_metrics(target.iloc[holdout], probabilities),
            }
        )
        if family == "logistic":
            joblib.dump(model, models / "logistic.joblib")
        elif group == "all_history":
            model.save_model(str(models / "catboost.cbm"))
            calibration_probabilities = model.predict_proba(
                cat_features.iloc[calibration]
            )[:, 1]
            calibrator = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)
            calibrator.fit(
                probability_logit(calibration_probabilities), target.iloc[calibration]
            )
            calibrated = calibrator.predict_proba(probability_logit(probabilities))[
                :, 1
            ]
            predictions["catboost_calibrated"] = calibrated
            holdout_scores.append(
                {
                    "model": "catboost_calibrated",
                    "features": len(columns),
                    "fit_seconds": None,
                    **probability_metrics(target.iloc[holdout], calibrated),
                }
            )
            joblib.dump(calibrator, models / "calibrator.joblib")
            pd.Series(model.feature_importances_, index=columns).sort_values(
                ascending=False
            ).rename("importance").to_csv(report / "feature_importance.csv")
    pd.DataFrame(holdout_scores).to_csv(report / "holdout_metrics.csv", index=False)
    prediction_table = data.iloc[holdout][["SK_ID_CURR", "TARGET"]].copy()
    for name, probabilities in predictions.items():
        prediction_table[name] = probabilities
    prediction_table.to_csv(report / "holdout_predictions.csv", index=False)
    segment_rows = []
    for name in ["catboost_all_history", "catboost_calibrated"]:
        probabilities = predictions[name]
        risk_deciles(target.iloc[holdout], probabilities).to_csv(
            report / f"{name}_deciles.csv", index=False
        )
        for flag in [column for column in features if column.startswith("HAS_")]:
            for value in [0, 1]:
                mask = features.iloc[holdout][flag].eq(value).to_numpy()
                segment_rows.append(
                    {
                        "model": name,
                        "history": flag,
                        "present": value,
                        "clients": int(mask.sum()),
                        "default_rate": target.iloc[holdout][mask].mean(),
                        **probability_metrics(
                            target.iloc[holdout][mask], probabilities[mask]
                        ),
                    }
                )
    pd.DataFrame(segment_rows).to_csv(report / "history_segments.csv", index=False)
    (report / "experiment.json").write_text(
        json.dumps(
            {
                "seed": 42,
                "cv_folds": 3,
                "catboost_iterations": 500,
                "catboost_depth": 6,
                "catboost_learning_rate": 0.05,
                "logistic_C": float(best_c),
                "platform": platform.platform(),
                "train_rows": len(train),
                "calibration_rows": len(calibration),
                "holdout_rows": len(holdout),
                "calibration_slope": float(calibrator.coef_[0, 0]),
                "calibration_intercept": float(calibrator.intercept_[0]),
            },
            indent=2,
        )
    )
    print(pd.DataFrame(holdout_scores).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
