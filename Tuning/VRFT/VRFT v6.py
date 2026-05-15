# Run script to optimise PID parameters. The script deploys kp, ki and kd for each electrode to PID_params.csv, 
# and generates a 200s 1.2mOhm reference signal.
# The script also generates an open_loop_params.csv file.
import numpy as np
from scipy.signal import lfilter
from numpy.linalg import lstsq
from scipy import signal
from scipy.linalg import block_diag

import scipy as sp
import sympy as sm
import pandas as pd
import matplotlib.pyplot as plt
import pysindy as ps


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

# Path to fusion
fusion_path = project_root / "fusion"

# Move working directory to meta_arx
os.chdir(project_root)
# Insert new working directory into PATH; known issue in Vscode
if str(module_path) not in sys.path:
    sys.path.insert(0,str(module_path))

from fusion.run_closed_loop import run_closed_loop_from_config
os.chdir(module_path)
#%%
# ----------------------------
# Define hyperparameters
# -----------------------------

# Specify a reference closed loop transfer function on the form:
# e^(-tau*s)/(1+0.2*t*s)^q
tau = 0   # Time delay
t = 30    # Settling time for the system poles
q = 3     # System order
N = 2000  # Simulation time. Used only in the data generation step
Ts = 1    # Discretisation interval

A = 2     # Amplitude of random data generation. Effectively a hyperparameter.

# Specify a frequency weighting function on the form:
# omega/(omega+s)
omega=0.1 # Cutoff frequency in the frequency weighting function

testing = True # Runs the Simulator at the end of the script with the new controller if True.
# NOTE: Testing must be set to false if q>3, due to lack of initial conditions.

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
    # f, pxx = sp.signal.welch(u)
    # G = [] # Initialise phi^-1/2
    # for i in pxx:
    #     G.append(1/np.sqrt(i))
    # Phi_inv_num = np.polyfit(f,G,1) # Fit a linear curve to G
    # For now ignore the spectral power component
    Phi_inv_num = [1] 
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
def generate_reference(N,method = "linear",amp = 2,ss = 1.2):
    if method == "linear":
        r=ss*np.ones(N)
    if method == "stair":
        r = np.zeros(N)
        for i in range(0,N):
            if i<N/(4*Ts):
                r[i] = 1
            elif i >= N/(4*Ts) and i < N/(2*Ts):
                r[i] = 2
            elif i >= N/(2*Ts) and i < (3*N)/(4*Ts):
                r[i] = 3
            elif i >= (3*N)/(4*Ts):
                r[i] = 0.5
    if method == "random":
        r = np.random.random(N)*amp
    data = pd.DataFrame(r, columns=["r"])
    save_location = project_root / "meta_arx" / "run_simulation" / "init_data" / "reference.csv"
    data.to_csv(save_location, index=False)

# The following function reads the output file from a simulator run
def read_output():
    output_location = project_root / "fusion" / "run_simulation" / "history" / "closed_loop_sim.csv"
    data = pd.read_csv(output_location)
    t = data["t_s"]
    y = np.column_stack([data["y1"],data["y2"],data["y3"]])
    u = np.column_stack([data["u1"],data["u2"],data["u3"]])
    e = np.column_stack([data["e1"],data["e2"],data["e3"]])
    r = np.column_stack([data["r1"],data["r2"],data["r3"]])
    GP_var = np.column_stack([data["gp_var1"],data["gp_var2"],data["gp_var3"]])
    return t, y, u, e, r, GP_var

#%%
# -----------------------------
# Data collection is preformed below. The simulator is ran in open-loop with a continually exciting input
# -----------------------------

# Deploy a dummy parameter for open loop run.
theta_PID = np.array(([1]))
data = pd.DataFrame(theta_PID, columns=["u_constant"])
save_location = project_root / "meta_arx" / "run_simulation" / "init_data" / "open_loop_params.csv"
save_location_velocity = project_root / "meta_arx" / "run_simulation" / "init_data" / "open_loop_velocity_params.csv"
data.to_csv(save_location, index=False)
data.to_csv(save_location_velocity, index=False)
# Generate a white noise referense (continually exciting across all frequencies)
generate_reference(N,"random",A)

# Run simulation with open-loop controller:
run_closed_loop_from_config(
    ref_csv="run_simulation/init_data/reference.csv",
    controller_name="open_loop_velocity",
    controller_config=save_location_velocity,
    out_csv="run_simulation/history/closed_loop_sim.csv",
    dt=1.0,
)

#%% TEMP TEST DELETE CELL WHEN DONE
# save_locationPID = project_root / "meta_arx" / "run_simulation" / "init_data" / "PID_params.csv"
# ttt = pd.read_csv(save_locationPID)

# names = list(ttt.columns)
# arr = ttt.to_numpy()
# lin1 = arr[0,:]
# lin2 = arr[1,:]
# lin3 = arr[2,:]
# coeffs = block_diag(lin1,lin2,lin3)
# print(coeffs)
# print(ttt.columns)

ffh = np.array([1,2,3])
ffg = ffh*2
print(ffg)

#%%
t_data, y ,u_pos, _, r, _ = read_output()

u = np.gradient(u_pos,axis=0)

#%%
# -----------------------------
# Construct virtual reference and error
# -----------------------------

# Construct the discrete time numerator and denominator of the reference model
M_num, M_den = M_cont_to_disc(tau,t,q,Ts)
# Construct the discrete time numerator and denominator of frequency weighting function
W_num, W_den=W_cont_to_disc(omega,Ts)
# Fit a model to the spectral power of the inpuits
Phi1_num, Phi1_den = construct_phi(u[:,0])
Phi2_num, Phi2_den = construct_phi(u[:,1])
Phi3_num, Phi3_den = construct_phi(u[:,2])
# Construct the main VRFT filter
F1_num, F1_den, aux1_num, aux1_den = GetFilterCoeff(M_num,M_den,W_num,W_den,Phi1_num,Phi1_den)
F2_num, F2_den, aux2_num, aux2_den = GetFilterCoeff(M_num,M_den,W_num,W_den,Phi2_num,Phi2_den)
F3_num, F3_den, aux3_num, aux3_den = GetFilterCoeff(M_num,M_den,W_num,W_den,Phi3_num,Phi3_den)

# Filter y through the VRFT filter
y_l1 = lfilter(F1_num,F1_den,y[:,0],axis=0)
y_l2 = lfilter(F2_num,F2_den,y[:,1],axis=0)
y_l3 = lfilter(F3_num,F3_den,y[:,2],axis=0)
# Construct the virtual error
e_l1 = lfilter(aux1_num,aux1_den,y[:,0],axis=0) - y_l1
e_l2 = lfilter(aux2_num,aux2_den,y[:,1],axis=0) - y_l2
e_l3 = lfilter(aux3_num,aux3_den,y[:,2],axis=0) - y_l3
# Filtered input
u_l1 = lfilter(F1_num, F1_den, u[:,0])
u_l2 = lfilter(F2_num, F2_den, u[:,1])
u_l3 = lfilter(F3_num, F3_den, u[:,2])
#%%
# -----------------------------
# Define controller structure C(z, θ), in this case a PID controller
# -----------------------------
# Implement a PID controller:
# The derivative term is implemented as the backwards difference.
# The integral terms is implemented as the cumulative sum of all error terms.
# C(z,θ): u = kp * e[k] + ki * sum(e[k])*Ts - kd * (y[k]-y[k-1])/Ts
phi1 = np.column_stack([e_l1[1:],np.cumsum(e_l1)[1:]*Ts,-(y_l1[1:]-y_l1[:-1])/Ts])
phi2 = np.column_stack([e_l2[1:],np.cumsum(e_l2)[1:]*Ts,-(y_l2[1:]-y_l2[:-1])/Ts])
phi3 = np.column_stack([e_l3[1:],np.cumsum(e_l3)[1:]*Ts,-(y_l3[1:]-y_l3[:-1])/Ts])
#%%
# -----------------------------
# Calculate PID params and deploy
# -----------------------------



# Solve VRFT optimisation problem with an OLS approach.
theta_PID, _, _, _ = lstsq(block_diag(phi1,phi2,phi3), np.concatenate([u_l1[:-1],u_l2[:-1],u_l3[:-1]]), rcond=None)
# Clamp negative values for the integrator term.
#theta_PID[1],theta_PID[4],theta_PID[7] = 0,0,0


print("Tuned controller parameters (θ):", theta_PID)

# Deploy PID params
data = pd.DataFrame(theta_PID.reshape(3,3), columns=["kp","ki","kd"])
save_location = project_root / "meta_arx" / "run_simulation" / "init_data" / "PID_params.csv"
data.to_csv(save_location, index=False)


theta_test,_,_,_ = lstsq(phi1,u_l1[:-1],rcond=None)
print("testparameters: ",theta_test)
#%% SINDy solution

e_l1 = e_l1.values if hasattr(e_l1, "values") else e_l1
e_l2 = e_l2.values if hasattr(e_l2, "values") else e_l2
e_l3 = e_l3.values if hasattr(e_l3, "values") else e_l3

u_l1 = u_l1.values if hasattr(u_l1, "values") else u_l1
u_l2 = u_l2.values if hasattr(u_l2, "values") else u_l2
u_l3 = u_l3.values if hasattr(u_l3, "values") else u_l3

t_data = t_data.values if hasattr(t_data, "values") else t_data


Libraries = [ps.PolynomialLibrary(),ps.FourierLibrary()]

Lib = ps.GeneralizedLibrary(Libraries)
Lib = ps.PolynomialLibrary(degree=1)
opt = ps.STLSQ(threshold=0.0001) # Use sequentially thresholded least squares

e1 = np.column_stack([e_l1[1:],-(y_l1[1:]-y_l1[:-1])/Ts])
e2 = np.column_stack([e_l2[1:],-(y_l2[1:]-y_l2[:-1])/Ts])
e3 = np.column_stack([e_l3[1:],-(y_l3[1:]-y_l3[:-1])/Ts])

e_comb = np.column_stack([e_l1[1:],-(y_l1[1:]-y_l1[:-1])/Ts,
                          e_l2[1:],-(y_l2[1:]-y_l2[:-1])/Ts,
                          e_l3[1:],-(y_l3[1:]-y_l3[:-1])/Ts])

X = np.column_stack([e1,e2,e3])

X_dot = np.column_stack([u_l1[1:],u_l2[1:],u_l3[1:]])

feature_names = ["e1","y1_d","e2","y2_d","e3","y3_d"]


model1 = ps.SINDy(optimizer=opt,feature_library=Lib)
model1.fit(X, x_dot=u_l1[1:],t=Ts,feature_names=feature_names)   
model1.print(precision=8)

model2 = ps.SINDy(optimizer=opt,feature_library=Lib)
model2.fit(X, x_dot=u_l2[1:],t=Ts,feature_names=feature_names)  
model2.print(precision=8)

model3 = ps.SINDy(optimizer=opt,feature_library=Lib)
model3.fit(X, x_dot=u_l3[1:],t=Ts,feature_names=feature_names)  
model3.print(precision=8)



#model1.score(X,Ts,X_dot)
model = ps.SINDy(optimizer=opt,feature_library=Lib)
model.fit(X, x_dot=X_dot,t=Ts,feature_names=feature_names)   
model.print(precision=8)

xx = model.coefficients()
#%% Iterative solving for sparsity coefficient



threshold_scan = np.linspace(0,0.001,25)
coeffs = []

for i, threshold in enumerate(threshold_scan):
    opt = ps.STLSQ(threshold=threshold)
    model = ps.SINDy(optimizer=opt,feature_library=Lib)
    model.fit(X, x_dot=X_dot,t=Ts,feature_names=feature_names)
    coeffs.append(model.score(X,Ts,X_dot))

plt.plot(threshold_scan,coeffs)
plt.show()


#%% 
# -----------------------------
# Testing. Commented out by default.
# -----------------------------

if testing == True:
    generate_reference(400,method="linear",ss=1.2)
    run_closed_loop_from_config(
        ref_csv="run_simulation/init_data/reference.csv",
        controller_name="generalized_controller",
        controller_config="run_simulation/init_data/generalized_params.csv",
        out_csv="run_simulation/history/closed_loop_sim.csv",
        dt=1.0,
        )
    
    t_test,y_test,u_test,_,r_test,gp_var_test = read_output()
    plt.figure()
    plt.subplot(311)
    plt.plot(t_test,y_test[:,0],label="el1 resistance")
    plt.plot(t_test,r_test[:,0],"k:",lw=1)
    plt.legend()
    plt.tick_params('x', labelbottom=False)
    
    plt.subplot(312)
    plt.plot(t_test,y_test[:,1],"r",label="el2 resistance")
    plt.plot(t_test,r_test[:,1],"k:",lw=1)
    plt.legend()
    plt.tick_params('x', labelbottom=False)
    
    plt.subplot(313)
    plt.plot(t_test,y_test[:,2],"g",label="el3 resistance")
    plt.plot(t_test,r_test[:,2],"k:",lw=1)
    plt.legend()

    plt.show()
    
    plt.figure()
    plt.subplot(311)
    plt.plot(t_test,u_test[:,0],label="el1 position")
    plt.legend()
    plt.tick_params('x', labelbottom=False)
    
    plt.subplot(312)
    plt.plot(t_test,u_test[:,1],"r",label="el2 position")
    plt.legend()
    plt.tick_params('x', labelbottom=False) 
    
    plt.subplot(313)
    plt.plot(t_test,u_test[:,2],"g",label="el3 position")
    plt.legend()
    
    plt.show()

    plt.plot(t_test,gp_var_test[:,0],"b",label = "El1 GP variance")
    plt.plot(t_test,gp_var_test[:,1],"r",label = "El2 GP variance")   
    plt.plot(t_test,gp_var_test[:,2],"g",label = "El3 GP variance")
    plt.legend()
    plt.show()
    
# Clean up refernce; restores 200 second 1.2 mOhm reference:
generate_reference(200)
