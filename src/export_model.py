import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import joblib
import os

def export_model():
    print("Generating simulated EEG and Video features to train the fallback model...")
    np.random.seed(42)
    n_samples = 100

    # EEG features
    eeg_features = pd.DataFrame({
        "eeg_alpha_mean": np.random.normal(0, 1, n_samples),
        "eeg_beta_mean":  np.random.normal(0, 1, n_samples),
        "eeg_theta_mean": np.random.normal(0, 1, n_samples),
        "eeg_corr_mean":  np.random.uniform(0.2, 0.8, n_samples),
        "eeg_corr_var":   np.random.uniform(0.01, 0.2, n_samples),
    })

    # Video features
    video_features = pd.DataFrame({
        "mouth_motion_var": np.random.uniform(0.0, 1.0, n_samples),
        "eye_motion_var":   np.random.uniform(0.0, 1.0, n_samples),
        "head_motion_var":  np.random.uniform(0.0, 1.0, n_samples),
    })

    labels = pd.DataFrame({
        "label": np.random.randint(0, 2, n_samples)
    })

    X = pd.concat([eeg_features, video_features], axis=1)
    y = labels["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(class_weight="balanced", solver="liblinear"))
    ])

    model.fit(X_train, y_train)
    print("Model trained.")

    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/model.joblib")
    print("Model successfully exported to models/model.joblib")

if __name__ == "__main__":
    export_model()
