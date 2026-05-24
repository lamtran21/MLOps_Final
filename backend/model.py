import pickle
import numpy as np
from sklearn.metrics import mean_squared_error
from pathlib import Path

MODEL_PATH = Path(__file__).parent.parent / "models" / "champion_model.pkl"
NON_FEATURE_COLS = {"median_house_value", "TARGET", "prediction"}


def load_model():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


def predict(model, df):
    feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLS]
    if not feature_cols:
        raise ValueError("No feature columns found in uploaded CSV")
    X = df[feature_cols]
    return model.predict(X)

def compute_rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))