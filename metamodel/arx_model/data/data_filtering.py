import pandas as pd
from scipy.signal import butter, filtfilt

df = pd.read_csv("metamodel/arx_model/data/cleaned_data/07_08_1s_residuals.csv", parse_dates=['timestamp']).set_index('timestamp')

def butter_lowpass_series(s: pd.Series, fc_hz: float, fs_hz: float, order=4):
    nyq = 0.5 * fs_hz
    Wn = fc_hz / nyq
    b, a = butter(order, Wn, btype='low', analog=False)
    x = s.interpolate(limit_direction='both').astype(float)
    y = filtfilt(b, a, x.values)
    return pd.Series(y, index=s.index, name=s.name)

# --- determine sample rate automatically ---
freq = pd.infer_freq(df.index) or '30S'
dt = pd.to_timedelta(freq).total_seconds()
fs = 1.0 / dt

# pick cutoff (fc) based on sampling
fc = 0.01 if fs < 0.1 else 0.15  # e.g. 0.01Hz ≈ 100s period for 30s data, 0.15Hz ≈ 6.7s for 1s data

# --- filter residual and key exogenous signals ---
df['residual_filt'] = butter_lowpass_series(df['residual'], fc_hz=fc, fs_hz=fs, order=2)

for c in ['El1_kA', 'El2_kA', 'El3_kA', 'Tap_A', 'Tap_B', 'Tap_C', 'RMS_V_transformer']:
    if c in df.columns:
        df[c + '_filt'] = butter_lowpass_series(df[c], fc_hz=fc, fs_hz=fs, order=4)

print(df.filter(like='filt').head())
output_path = "metamodel/arx_model/data/cleaned_data/07_08_1s_filtered_data.csv"
df.to_csv(output_path)
print(f"[saved] → {output_path}")
