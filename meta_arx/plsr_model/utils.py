import numpy as np
from itertools import combinations
import pandas as pd

def cols_by_tag(df, tags):
    """
    Return the column names whose header contains *any* of the given tags
    after the first comma. Case-insensitive.
    - tags: str or list of str, e.g. "ge63" or ["ge63","ge64"]
    This seems to work well for the current naming scheme in the excel file.
    """
    if isinstance(tags, str):
        tags = [tags]
    want = {t.strip().lower() for t in tags}

    matched = []
    for col in map(str, df.columns):
        parts = [p.strip() for p in col.split(",")]
        tokens_after_comma = parts[1:] if len(parts) > 1 else []
        tokens_lower = {t.lower() for t in tokens_after_comma}
        if want & tokens_lower:
            matched.append(col)
    return matched

def get_col_by_tag(df, tag):
    """
    Return a Series by tag after the comma.
    """
    hits = cols_by_tag(df, tag)

    return df[hits[0]]


def build_xis(df: pd.DataFrame, cols: list[str]) -> tuple[np.ndarray, list[str]]:
    "Automatically build interaction and square terms from base columns. Names for later ..."

    X_list, names = [], []
    # linear
    for c in cols:
        X_list.append(df[c].astype(float).to_numpy()[:, None])
        names.append(c)
    # interactions
    for a, b in combinations(cols, 2):
        X_list.append((df[a].to_numpy() * df[b].to_numpy())[:, None])
        names.append(f"{a}*{b}")
    # squares
    for c in cols:
        X_list.append((df[c].to_numpy() ** 2)[:, None])
        names.append(f"{c}**2")
    X = np.hstack(X_list)
    
    return X, names