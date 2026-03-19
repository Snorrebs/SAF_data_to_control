# VRFT v5; implemented spectral power component and improved integration with simulator.
# Run script to optimise PID parameters. The script deploys kp, ki and kd to PID_params.csv, 
# and generates a 200s 1.2mOhm reference signal.
# The script also generates an open_loop_params.csv file.
import numpy as np
from scipy.signal import lfilter
from numpy.linalg import lstsq
from scipy import signal

import scipy as sp
import sympy as sm
import pandas as pd
import matplotlib.pyplot as plt

import os
from pathlib import Path 

import sys
# Clear imported simulator modules to avoid issues with python cache.
for m in list(sys.modules):
    if m.startswith("run_simulation"):
        del sys.modules[m]

# Relative path of this script
VRFTuning = Path(__file__).resolve()

# project root path
project_root = VRFTuning.parent.parent.parent

# Path to meta_arx
module_path = project_root / "meta_arx"

# Move working directory to meta_arx
os.chdir(module_path)

from run_simulation.scripts.run_closed_loop import run_closed_loop_from_config

#%%
# ----------------------------
# Define hyperparameters
# -----------------------------

# Specify a reference closed loop transfer function on the form:
# e^(-tau*s)/(1+0.2*t*s)^q
tau = 0   # Time delay
t = 20    # Settling time for the system poles
q = 3     # System order
N = 2000  # Simulation time. Used only in the data generation step
Ts=1      # Discretisation interval

A = 2     # Amplitude of random data generation. Effectively a hyperparameter.

# Specify a frequency weighting function on the form:
# omega/(omega+s)
omega=0.1 # Cutoff frequency in the frequency weighting function

# No need to interact with anything else in this script for simple tuning. Some functions defined below may be usefull however.
#%%
# ----------------------------
# Functions related to VRFT.
# -----------------------------
# Convert the reference transfer function to discrete time:
def M_cont_to_disc(tau,t,q,Ts):
    den_coeff=np.polynomial.polynomial.polypow([1,0.2*t],q)
    den_coeff=list(reversed(den_coeff)) # Polypow takes coefficients in ascending order, scipy.signal related functions takes them in descending order.
    num_coeff=[1] # Initialise numerator. time delays are added in discrete time.
    M_CT=signal.TransferFunction(num_coeff,den_coeff)
    M_DT=M_CT.to_discrete(Ts, method="bilinear")
    delay_samples = int(round(tau / Ts))
    if delay_samples > 0:
    # Multiply numerator by z^-d (pad with zeros)
        M_DT_num = np.concatenate([np.zeros(delay_samples), (M_DT.num)])
    else:
        M_DT_num = M_DT.num
    M_DT_den = M_DT.den
    return M_DT_num, M_DT_den

# Convert the frequency weighting function to discrete time:
def W_cont_to_disc(omega,Ts):
    W_CT = signal.TransferFunction([omega], [1, omega])
    W_DT = W_CT.to_discrete(Ts, method='bilinear')
    return W_DT.num, W_DT.den

# Construct phi^-1/2. Use a linear fit; will be improved in the future.
def construct_phi(u):
    f, pxx = sp.signal.welch(u)
    G = [] # Initialise phi^-1/2
    for i in pxx:
        G.append(1/np.sqrt(i))
    Phi_inv_num = np.polyfit(f,G,1) # Fit a linear curve to G
    Phi_inv_den = [1] 
    return Phi_inv_num, Phi_inv_den

# Function for constructing the main VRFT filter.
def GetFilterCoeff(num,den,lp_num,lp_den,Phi_inv_num,Phi_inv_den):
    x=sm.symbols("x")
    def subConstructPoly(coeff,var=x):
        # Constructs the polynomial coeff[0]+coeff[1]*x+coeff[2]x**2...
        deg=len(coeff)-1
        return sum(c*var**(deg-i) for i,c in enumerate(reversed(coeff)))
    
    def subConstructRational(num,den):
        # Constructs the reference transfer function for internal use
        return subConstructPoly(num)/subConstructPoly(den)
        
    def subGetCoeffs(expr,var=x):
        # Extracts the coefficients of the filter for use with scipy lfilter
        num, den = sm.fraction(sm.simplify(expr))
        num_coeffs = sm.Poly(num, var).all_coeffs()
        den_coeffs = sm.Poly(den, var).all_coeffs()
            
        return list(reversed(num_coeffs)), list(reversed(den_coeffs))
    
    
    M = subConstructRational(num,den) # Construct reference transfer function
    W = subConstructRational(lp_num,lp_den) # Construct Lowpass filter
    Phi_Sqrt_inv = subConstructRational(Phi_inv_num,Phi_inv_den) # Construct Phi^(-1/2)
    
    F = M * (1 - M) * W * Phi_Sqrt_inv# Construct filter used on e_vr(t) and u(t)
    F_num, F_den = subGetCoeffs(F) 
    F_aux = (1 - M) * W * Phi_Sqrt_inv # Auxiliary filter to avoid having to calculate r_v directly, thus avoiding anti-causal filtering.
    aux_num, aux_den = subGetCoeffs(F_aux)
      
    # Convert from sympy float to numpy float
    
    F_num = np.array([float(i) for i in F_num], dtype=float)
    F_den = np.array([float(i) for i in F_den], dtype=float)
    aux_num = np.array([float(i) for i in aux_num], dtype=float)
    aux_den = np.array([float(i) for i in aux_den], dtype=float)
    return F_num, F_den, aux_num, aux_den

#%%
# ----------------------------
# The following are convenience functions for interacting with the simulator
# -----------------------------

# The following function is a reference generator with some options.
def generate_reference(N,method = "linear",amp = 2):
    if method == "linear":
        r=1.2*np.ones(N)
    if method == "stair":
        r = np.zeros(N)
        for i in range(0,N):
            if i<25/Ts:
                r[i] = 1
            elif i >= 25/Ts and i < 50/Ts:
                r[i] = 2
            elif i >= 50/Ts and i < 75/Ts:
                r[i] = 3
            elif i >= 75/Ts:
                r[i] = 0
    if method == "random":
        r = np.random.random(N)*amp
    data = pd.DataFrame(r, columns=["r"])
    save_location = project_root / "meta_arx" / "run_simulation" / "init_data" / "reference.csv"
    data.to_csv(save_location, index=False)

# The following function reads the output file from a simulator run
def read_output():
    output_location = project_root / "meta_arx" / "run_simulation" / "history" / "closed_loop_sim.csv"
    data = pd.read_csv(output_location)
    t = data["t_s"]
    y = data["y_pred"]
    u = data["u_cmd"]
    e = data["error"]
    r = data["reference"]
    return t, y, u, e, r

#%%
# -----------------------------
# Data collection is preformed below. The simulator is ran in open-loop with a continually exciting input
# -----------------------------

# Deploy a dummy parameter for open loop run.
theta_PID = np.array(([1]))
data = pd.DataFrame(theta_PID, columns=["u_constant"])
save_location = project_root / "meta_arx" / "run_simulation" / "init_data" / "open_loop_params.csv"
data.to_csv(save_location, index=False)

# Generate a white noise referense (continually exciting across all frequencies)
generate_reference(N,"random",1)

# Run simulation with open-loop controller:
run_closed_loop_from_config(
    ref_csv="run_simulation/init_data/reference.csv",
    controller_name="open_loop",
    controller_config="run_simulation/init_data/open_loop_params.csv",
    out_csv="run_simulation/history/closed_loop_sim.csv",
    dt=1.0,
)

_, y ,u_pos, _, _ = read_output()

#u = np.gradient(u_pos) <- this term requires more work
u = u_pos
# Clean up refernce; restores 200 second 1.2 mOhm reference:
generate_reference(200)

#%%
# -----------------------------
# Construct virtual reference and error
# -----------------------------
# r_v = M^-1(z) y
# First invert M (apply filter defined by denominator/ numerator swapped)
# That is, r_v = lfilter(M_den, M_num, y)

M_num, M_den = M_cont_to_disc(tau,t,q,Ts)

lp_num, lp_den=W_cont_to_disc(omega,Ts)

Phi_inv_num, Phi_inv_den = construct_phi(u)

F_num, F_den, aux_num, aux_den = GetFilterCoeff(M_num,M_den,lp_num,lp_den,Phi_inv_num,Phi_inv_den)

y_v = lfilter(F_num,F_den,y)

# Virtual, filtered error
e_v = lfilter(aux_num,aux_den,y) - y_v
# Filtered input
u_l=lfilter(F_num, F_den, u)
#%%
# -----------------------------
# Define controller structure C(z, θ), in this case a PID controller
# -----------------------------
# Implement a PID controller:
# The derivative term is implemented as the backwards difference.
# The integral terms is implemented as the cumulative sum of all error terms.
# C(z,θ): u = kp * e[k] + ki * sum(e[k])*Ts - kd * (y[k]-y[k-1])/Ts
phi_PID=np.column_stack([e_v[1:],np.cumsum(e_v)[1:]*Ts,-(y_v[1:]-y_v[:-1])/Ts])
#%%
# -----------------------------
# Calculate PID params and deploy
# -----------------------------

# Solve VRFT optimisation problem with an OLS approach.
theta_PID, _, _, _ = lstsq(phi_PID, u_l[1:], rcond=None)

print("Tuned controller parameters (θ):", theta_PID)

# Deploy PID params
data = pd.DataFrame(theta_PID.reshape(1,3), columns=["kp","ki","kd"])
save_location = project_root / "meta_arx" / "run_simulation" / "init_data" / "PID_params.csv"
data.to_csv(save_location, index=False)

#%% 
# -----------------------------
# Testing. Commented out by default.
# -----------------------------

# run_closed_loop_from_config(
#     ref_csv="run_simulation/init_data/reference.csv",
#     controller_name="pid",
#     controller_config="run_simulation/init_data/PID_params.csv",
#     out_csv="run_simulation/history/closed_loop_sim.csv",
#     dt=1.0,
# )

# t_test, y_test ,u_test, _, REF_test = read_output()

# plt.plot(t_test,y_test,label="resistance")
# plt.plot(t_test,REF_test,label="reference")
# plt.plot(t_test,u_test,label="electrode position")
# plt.legend()
# plt.show()
