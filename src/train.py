"""เทรนโมเดลสุดท้ายซ้ำได้ด้วยคำสั่งเดียว: python src/train.py

ทำขั้นตอนเดียวกับ notebooks/02_model.ipynb (เฉพาะโมเดลที่เลือกแล้ว)
แล้วบันทึกไฟล์ models/stroke_model.joblib ให้เว็บ Flask ใช้
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, classification_report,
                             confusion_matrix, precision_recall_curve, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "healthcare-dataset-stroke-data.csv"
MODEL_PATH = ROOT / "models" / "stroke_model.joblib"

RANDOM_STATE = 42
TARGET_RECALL = 0.80
NUM_COLS = ["age", "avg_glucose_level", "bmi"]
BIN_COLS = ["hypertension", "heart_disease"]
CAT_COLS = ["smoking_status"]
FEATURES = ["age", "hypertension", "heart_disease", "avg_glucose_level", "bmi", "smoking_status"]


def load_data():
    df = pd.read_csv(DATA_PATH)
    df = df[df["gender"] != "Other"]          # มีแค่ 1 แถว เรียนรู้ไม่ได้
    return df[FEATURES], df["stroke"]


def build_model():
    preprocessor = ColumnTransformer([
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("scaler", StandardScaler()),
        ]), NUM_COLS),
        ("bin", "passthrough", BIN_COLS),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_COLS),
    ])
    return Pipeline([
        ("prep", preprocessor),
        ("model", LogisticRegression(C=0.1, l1_ratio=0, solver="liblinear",
                                     class_weight="balanced", max_iter=1000)),
    ])


def choose_threshold(model, X_train, y_train):
    """เลือก threshold สูงสุดที่ยังได้ Recall >= TARGET_RECALL จาก out-of-fold prediction"""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    oof = cross_val_predict(model, X_train, y_train, cv=cv, method="predict_proba")[:, 1]
    precision, recall, thresholds = precision_recall_curve(y_train, oof)
    ok = thresholds[recall[:-1] >= TARGET_RECALL]
    return float(np.floor(ok.max() * 100) / 100)


def main():
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)
    print(f"Train {X_train.shape} | Test {X_test.shape}")

    model = build_model()
    threshold = choose_threshold(model, X_train, y_train)
    model.fit(X_train, y_train)

    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= threshold).astype(int)
    print(f"Threshold = {threshold}")
    print(classification_report(y_test, pred, target_names=["No stroke", "Stroke"], digits=3))
    print("Confusion matrix:\n", confusion_matrix(y_test, pred))
    print(f"PR-AUC {average_precision_score(y_test, proba):.4f} | ROC-AUC {roc_auc_score(y_test, proba):.4f}")

    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump({"model": model, "threshold": threshold, "features": FEATURES,
                 "sklearn_version": sklearn.__version__}, MODEL_PATH)
    print("บันทึกแล้ว:", MODEL_PATH.relative_to(ROOT))


if __name__ == "__main__":
    main()
