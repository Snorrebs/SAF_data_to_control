#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from joblib import load
from pathlib import Path


# ----------------------------------------------------------
# CONFIG
# ----------------------------------------------------------
MODEL_NAME = "arx_el1res_5543_07_no_ridge"   # change to your model
MODEL_PATH = Path("arx/models/model_meta") / f"{MODEL_NAME}.meta.joblib"


def extract_ar_coeffs(bundle):
    """
    Extract AR coefficients from the model bundle.
    AR columns are those starting with 'y_filt_lag'.
    Returns coefficients in order [a1, a2, ..., ap].
    """
    model = bundle["model"]
    X_cols = bundle["X_cols"]

    ar_terms = []
    ar_coefs = []

    for name, coef in zip(X_cols, model.coef_):
        if "y_filt_lag" in name:
            lag = int(name.split("lag")[-1])
            ar_terms.append((lag, coef))

    # Sort by lag number
    ar_terms = sorted(ar_terms, key=lambda x: x[0])

    # Extract just coefficients
    ar_coefs = [coef for _, coef in ar_terms]

    return np.array(ar_coefs)


def compute_poles(ar_coefs):
    """
    Build AR polynomial:
        y(t+1) = a1*y(t) + a2*y(t-1) + ... + ap*y(t-p)
    AR polynomial = 1 - a1*z^-1 - a2*z^-2 - ...
    """
    poly = np.r_[1, -ar_coefs]
    poles = np.roots(poly)
    return poles


def plot_poles(poles):
    """
    Plot poles in the complex plane with unit circle.
    """
    plt.figure(figsize=(6,6))

    # Unit circle
    theta = np.linspace(0, 2*np.pi, 400)
    plt.plot(np.cos(theta), np.sin(theta), "k--", label="Unit circle")

    # Poles
    plt.scatter(poles.real, poles.imag, color="red", s=80, label="AR poles")

    # Axes
    plt.axhline(0, color="black", linewidth=0.5)
    plt.axvline(0, color="black", linewidth=0.5)

    plt.xlabel("Real")
    plt.ylabel("Imaginary")
    plt.title("ARX Model Pole Plot")
    plt.grid(True)
    plt.axis("equal")
    plt.legend()
    plt.tight_layout()
    plt.show()


def main():
    # Load bundle
    bundle = load(MODEL_PATH)

    # Extract AR coefficients
    ar_coefs = extract_ar_coeffs(bundle)
    print("AR coefficients:", ar_coefs)

    # Compute poles
    poles = compute_poles(ar_coefs)
    print("Poles:", poles)

    # Plot
    plot_poles(poles)


if __name__ == "__main__":
    main()
