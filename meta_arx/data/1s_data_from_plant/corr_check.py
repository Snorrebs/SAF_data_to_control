import pandas as pd
import numpy as np
from pathlib import Path

IN_CSV = Path("meta_arx/data/1s_data_from_plant/07_24.csv")
df = pd.read_csv(IN_CSV)

# ---------------------------------
# Targets
# ---------------------------------
target_current = "El1_kA"
target_resistance = "El1_Resistance_mOhm"

# Drop timestamp for correlation
df = df.drop(columns=["timestamp"], errors="ignore")

# Keep only numeric columns
df = df.select_dtypes(include=[np.number]).copy()

# ---------------------------------
# Compute DELTAS
# ---------------------------------
df_delta = df.diff().add_prefix("d_")

# Remove first NaN row
df = df.iloc[1:].reset_index(drop=True)
df_delta = df_delta.iloc[1:].reset_index(drop=True)

# ---------------------------------
# Correlation (LEVELS)
# ---------------------------------
corr_current = df.corr()[target_current].drop(target_current)
corr_resistance = df.corr()[target_resistance].drop(target_resistance)

# ---------------------------------
# Correlation (DELTAS)
# ---------------------------------
corr_current_delta = df_delta.corr()["d_" + target_current].drop("d_" + target_current)
corr_resistance_delta = df_delta.corr()["d_" + target_resistance].drop("d_" + target_resistance)

# ---------------------------------
# Sort by absolute correlation
# ---------------------------------
corr_current_sorted = corr_current.reindex(corr_current.abs().sort_values(ascending=False).index)
corr_resistance_sorted = corr_resistance.reindex(corr_resistance.abs().sort_values(ascending=False).index)

corr_current_delta_sorted = corr_current_delta.reindex(
    corr_current_delta.abs().sort_values(ascending=False).index
)
corr_resistance_delta_sorted = corr_resistance_delta.reindex(
    corr_resistance_delta.abs().sort_values(ascending=False).index
)

# ---------------------------------
# Print results
# ---------------------------------
print("\n===== LEVEL CORRELATION WITH El1_kA =====")
print(corr_current_sorted)

print("\n===== LEVEL CORRELATION WITH El1_Resistance_mOhm =====")
print(corr_resistance_sorted)

print("\n===== DELTA CORRELATION WITH d_El1_kA =====")
print(corr_current_delta_sorted)

print("\n===== DELTA CORRELATION WITH d_El1_Resistance_mOhm =====")
print(corr_resistance_delta_sorted)