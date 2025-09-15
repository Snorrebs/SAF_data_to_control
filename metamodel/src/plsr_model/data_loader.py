from __future__ import annotations
import pandas as pd
from pathlib import Path
from typing import Optional, Sequence

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"

#Excel file loader.
def load_data_from_excel(
        filename: str,
        sheet: Optional[str] = None,
        usecols: Optional[Sequence[str]] = None,
    )-> pd.DataFrame:

    path = RAW_DIR / filename
    df = pd.read_excel(path, sheet_name=sheet, engine='openpyxl')

    if usecols:
        df = df[list(usecols)]

    return df

#Load database
def load_db(filename="Database_NTNU_FeSi_March2025_v2.xlsx", sheet=0):
    path = RAW_DIR / filename

    return pd.read_excel(path, sheet_name=sheet, engine="openpyxl")

#Load regression coefficients
def load_regcoeff(filename="regkoeff_NTNU_FeSi_March2025_v2.xlsx", sheet="regkoeff"):
    path = RAW_DIR / filename
    df = pd.read_excel(path, sheet_name=sheet, engine="openpyxl")

    return df.set_index(df.columns[0])  # first column = term names (B_0, ge1, ge1*ge2, ge1**2, ...)
