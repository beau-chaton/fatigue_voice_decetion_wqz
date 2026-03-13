import pandas as pd
import joblib
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, classification_report

CSV = "features_egemaps.csv"
TEST_SPEAKER = "bea"   # 先用 jenie 做测试；你可以换成 bea/josh/sam 轮流跑
MODEL_OUT = "sleepy_opensmile_logreg.joblib"

def main():
    df = pd.read_csv(CSV)
    y = df["label_sleepy"].astype(int).values
    speaker = df["speaker"].astype(str).values
    drop_cols = {"path", "path_3s", "speaker", "label_sleepy", "folder", "sr", "window_seconds"}
    feature_cols = [c for c in df.columns if c not in drop_cols]

    X = df[feature_cols].values

    train_idx = speaker != TEST_SPEAKER
    test_idx = speaker == TEST_SPEAKER

    if not np.any(test_idx):
        raise RuntimeError(f"No rows for TEST_SPEAKER={TEST_SPEAKER}. Available speakers: {sorted(set(speaker))}")

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=5000, class_weight="balanced")),
    ])

    model.fit(X[train_idx], y[train_idx])

    p_test = model.predict_proba(X[test_idx])[:, 1]  # P(Sleepy) = fatigue_score
    y_test = y[test_idx]

    print("TEST_SPEAKER:", TEST_SPEAKER)
    print("AUC:", roc_auc_score(y_test, p_test))
    print("AP :", average_precision_score(y_test, p_test))
    print(classification_report(y_test, (p_test >= 0.5).astype(int), digits=4))

    joblib.dump(
        {"model": model, "feature_cols": feature_cols, "test_speaker": TEST_SPEAKER},
        MODEL_OUT
    )
    print("Saved:", MODEL_OUT)

if __name__ == "__main__":
    main()