"""
Hook for the steady‑state metamodel (PLSR surrogate).
Implement `predict_meta(df_inputs)` to produce *unfiltered* meta outputs.
For now we provide a simple pass‑through / placeholder.
"""


import pandas as pd


# 👇 replace this with your actual metamodel call (PLSR or loaded coefficients)


def predict_meta(df_inputs: pd.DataFrame, y_names: list[str]) -> pd.DataFrame:
    # TODO: call your metamodel to produce meta predictions for the outputs `y_names`
    # placeholder: zeros with proper columns/index
    meta = pd.DataFrame(0.0, index=df_inputs.index, columns=[f"meta_{y}" for y in y_names])
    return meta