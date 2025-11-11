import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib
from pathlib import Path


def fit_save_scaler(df: pd.DataFrame, cols: list[str], models_dir: Path, name: str = "scalers.joblib") -> StandardScaler:
    scaler = StandardScaler(with_mean=True, with_std=True)
    scaler.fit(df[cols])
    models_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump({"cols": cols, "scaler": scaler}, models_dir / name)
    return scaler


def load_scaler(models_dir: Path, name: str = "scalers.joblib") -> tuple[list[str], StandardScaler]:
    blob = joblib.load(models_dir / name)
    return blob["cols"], blob["scaler"]


def transform(df: pd.DataFrame, cols: list[str], scaler: StandardScaler) -> pd.DataFrame:
    df_out = df.copy()
    df_out[cols] = scaler.transform(df_out[cols])
    return df_out