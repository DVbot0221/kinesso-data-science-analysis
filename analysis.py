"""Reproducible KINESSO propensity-modelling analysis."""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

RANDOM_STATE = 42
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
FIGURES = ROOT / "figures"
FIGURES.mkdir(exist_ok=True)

def preprocessor(X):
    cat = X.select_dtypes(include=["object"]).columns
    num = X.select_dtypes(exclude=["object"]).columns
    return ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat),
        ("num", "passthrough", num)
    ])

def run_models(X, y, dataset):
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=.20, random_state=RANDOM_STATE, stratify=y
    )
    scale_pos = np.bincount(ytr)[0] / np.bincount(ytr)[1]

    models = {
        "Decision Tree": Pipeline([
            ("prep", preprocessor(Xtr)),
            ("model", DecisionTreeClassifier(
                max_depth=8, min_samples_leaf=10,
                class_weight="balanced", random_state=RANDOM_STATE))
        ]),
        "XGBoost": Pipeline([
            ("prep", preprocessor(Xtr)),
            ("model", XGBClassifier(
                n_estimators=300, max_depth=4, learning_rate=.05,
                subsample=.8, colsample_bytree=.8,
                scale_pos_weight=scale_pos,
                objective="binary:logistic", eval_metric="logloss",
                random_state=RANDOM_STATE, n_jobs=2))
        ])
    }

    rows = []
    for name, model in models.items():
        model.fit(Xtr, ytr)
        pred = model.predict(Xte)
        prob = model.predict_proba(Xte)[:, 1]
        rows.append({
            "Dataset": dataset, "Model": name,
            "Precision": precision_score(yte, pred, zero_division=0),
            "Recall": recall_score(yte, pred, zero_division=0),
            "F1": f1_score(yte, pred, zero_division=0),
            "ROC-AUC": roc_auc_score(yte, prob)
        })
    return pd.DataFrame(rows), models

def feature_plot(model, title, filename):
    prep = model.named_steps["prep"]
    estimator = model.named_steps["model"]
    names = prep.get_feature_names_out()
    fi = pd.DataFrame({
        "Feature": names,
        "Importance": estimator.feature_importances_
    }).nlargest(10, "Importance")
    plt.figure(figsize=(8, 5.2))
    plt.barh(fi["Feature"][::-1], fi["Importance"][::-1])
    plt.xlabel("Feature importance")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(FIGURES / filename, dpi=200, bbox_inches="tight")
    plt.close()

# Online Shoppers
shop = pd.read_csv(DATA / "online_shoppers_intention.csv").drop_duplicates()
shop["Weekend"] = shop["Weekend"].astype(int)
shop["Revenue"] = shop["Revenue"].astype(int)
shop_results, shop_models = run_models(
    shop.drop(columns="Revenue"), shop["Revenue"], "Online Shoppers"
)

# Bank Marketing
bank = pd.read_csv(DATA / "bank-full.csv", sep=";").drop(columns=["duration"])
bank["y"] = bank["y"].map({"no": 0, "yes": 1})
bank_results, bank_models = run_models(
    bank.drop(columns="y"), bank["y"], "Bank Marketing"
)

results = pd.concat([shop_results, bank_results], ignore_index=True)
print(results.round(3).to_string(index=False))
results.to_csv(ROOT / "model_results.csv", index=False)

feature_plot(
    shop_models["XGBoost"],
    "Online Shoppers — Top XGBoost Features",
    "online_shoppers_xgboost_feature_importance.png"
)
feature_plot(
    bank_models["XGBoost"],
    "Bank Marketing — Top XGBoost Features",
    "bank_marketing_xgboost_feature_importance.png"
)
