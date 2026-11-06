"""Leakage-aware model training and evaluation."""
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, confusion_matrix, roc_curve, classification_report)
from data_pipeline import NUMERIC, CATEGORICAL, TARGET


def build_preprocessor(df):
    nums = [c for c in NUMERIC if c in df.columns]
    cats = [c for c in CATEGORICAL if c in df.columns]
    numeric_pipe = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    categorical_pipe = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                                 ("onehot", OneHotEncoder(handle_unknown="ignore"))])
    return ColumnTransformer([("numeric", numeric_pipe, nums), ("categorical", categorical_pipe, cats)])


def train_models(df, tune=False):
    X = df.drop(columns=[TARGET, "customerID"], errors="ignore")
    y = (df[TARGET] == "Yes").astype(int)
    if y.nunique() < 2:
        raise ValueError("Training requires examples of both Churn=Yes and Churn=No.")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42, stratify=y)
    results, models, predictions = [], {}, {}
    candidates = {
        "Logistic Regression": LogisticRegression(max_iter=1500, class_weight="balanced", random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=250, max_depth=None, min_samples_leaf=2,
                                                class_weight="balanced_subsample", random_state=42, n_jobs=-1),
        "Extra Trees": ExtraTreesClassifier(n_estimators=250, min_samples_leaf=2, class_weight="balanced",
                                            random_state=42, n_jobs=-1)
    }
    try:
        from xgboost import XGBClassifier
        negatives = max(1, int((y_train == 0).sum()))
        positives = max(1, int((y_train == 1).sum()))
        candidates["XGBoost"] = XGBClassifier(n_estimators=250, max_depth=4, learning_rate=0.05,
            subsample=0.9, colsample_bytree=0.9, scale_pos_weight=negatives/positives,
            eval_metric="logloss", random_state=42, n_jobs=-1)
    except ImportError:
        pass
    for name, estimator in candidates.items():
        pipe = Pipeline([("prep", build_preprocessor(X_train)), ("model", estimator)])
        if tune and name == "Logistic Regression":
            search = GridSearchCV(pipe, {"model__C": [.1, 1, 5]}, scoring="recall", cv=3, n_jobs=-1)
            search.fit(X_train, y_train)
            pipe = search.best_estimator_
        else:
            pipe.fit(X_train, y_train)
        prob = pipe.predict_proba(X_test)[:, 1]
        pred = (prob >= .5).astype(int)
        models[name] = pipe
        predictions[name] = {"prob": prob, "pred": pred}
        results.append({"Model": name, "Accuracy": accuracy_score(y_test, pred),
                        "Precision": precision_score(y_test, pred, zero_division=0),
                        "Recall": recall_score(y_test, pred, zero_division=0),
                        "F1": f1_score(y_test, pred, zero_division=0),
                        "ROC-AUC": roc_auc_score(y_test, prob)})
    metrics = pd.DataFrame(results).sort_values(["Recall", "ROC-AUC", "F1"], ascending=False).reset_index(drop=True)
    best_name = metrics.iloc[0]["Model"]
    package = {"models": models, "metrics": metrics, "best_name": best_name,
               "X_test": X_test, "y_test": y_test, "predictions": predictions,
               "feature_columns": list(X.columns), "target": TARGET}
    return package


def save_package(package, path="models/churn_models.joblib"):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(package, path)
