import pandas as pd
import joblib

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

CSV = "features_egemaps.csv"
MODEL_OUT = "sleepy_opensmile_logreg.joblib"

def main():
    df = pd.read_csv(CSV)

    y = df["label_sleepy"].astype(int).values

    drop_cols = {"path", "path_3s", "speaker", "label_sleepy", "folder", "sr", "window_seconds"}
    feature_cols = [c for c in df.columns if c not in drop_cols]

    X = df[feature_cols].values

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=5000, class_weight="balanced")),
    ])

    model.fit(X, y)

    joblib.dump(
        {"model": model, "feature_cols": feature_cols, "trained_on": "all_data"},
        MODEL_OUT
    )
    print(f"Trained on all rows: {len(df)}")
    print("Saved:", MODEL_OUT)

if __name__ == "__main__":
    main()