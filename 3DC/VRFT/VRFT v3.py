## IMPLEMENTED SECOND ORDER SYSTEM


import numpy as np
from scipy.signal import lfilter
from numpy.linalg import lstsq
from scipy import signal

import scipy as sp
import sympy as sm
import matplotlib.pyplot as plt


import pandas as pd

import warnings # Will find a better solution for this at some point.
warnings.filterwarnings(
    "ignore",
    message="Conversion of an array with ndim > 0 to a scalar is deprecated",
    category=DeprecationWarning
)

# -----------------------------
# STEP 1: Define reference model M(z)
# -----------------------------

#%% 

# Specify a reference closed loop transfer function on the form:
# e^(-tau*s)/(1+0.2*t*s)^q
tau=0  #time delay. ###FIXED KINDA, OLD MESSAGE NOTE: for any value other than 0, calculating the virtual reference does not work due to a bug in scipy.
t=5     #settling time for the system poles
q=3     #system order

Ts=0.1 #discretisation interval

# specify a frequency weighting function on the form:
# omega/(omega+s)
omega=10 #cutoff frequency in the frequency weighting function

# Convert the reference transfer function to discrete time:
def M_cont_to_desc(tau,t,q,Ts):
    den_coeff=np.polynomial.polynomial.polypow([1,0.2*t],q)
    den_coeff=list(reversed(den_coeff)) #Polypow takes coefficients in ascending order, scipy transfer function stuff takes them in descending order.
    num_coeff=[1] #initialise numerator. time delays are added in discrete time.
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
#convert the frequency weighting function to discrete time:
def W_cont_to_desc(omega,Ts):
    W_CT = signal.TransferFunction([omega], [1, omega])
    W_DT = W_CT.to_discrete(Ts, method='bilinear')
    return W_DT.num, W_DT.den



#%%
#main function for constructing the error filter
def GetFilterCoeff(num,den,lp_num,lp_den):
    x=sm.symbols("x")
    def subConstructPoly(coeff,var=x):
        #Constructs the polynomial coeff[0]+coeff[1]*x+coeff[2]x**2...
        deg=len(coeff)-1
        return sum(c*var**(deg-i) for i,c in enumerate(reversed(coeff)))
    
    def subConstructRational(num,den):
        #Constructs the reference transfer function for internal use
        return subConstructPoly(num)/subConstructPoly(den)
        
    def subGetCoeffs(expr,var=x):
        #extracts the coefficients of the filter for use with scipy lfilter
        num, den = sm.fraction(sm.simplify(expr))
        num_coeffs = sm.Poly(num, var).all_coeffs()
        den_coeffs = sm.Poly(den, var).all_coeffs()
            
        return list(reversed(num_coeffs)), list(reversed(den_coeffs))
    
    
    M = subConstructRational(num,den) # Construct reference transfer function
    W = subConstructRational(lp_num,lp_den) # Construct Lowpass filter
    F = M * (1 - M) * W / 2 # construct filter used on e_vr(t) and u(t)
    F_num, F_den = subGetCoeffs(F) #Get coefficients of the filter
    F_aux = (1 - M) * W / 2 #auxiliary filter to avoid having to calculate r_v directly, thus circumventing the scipy bug.
    aux_num, aux_den = subGetCoeffs(F_aux)
      
    #convert from sympy float to numpy float
    
    F_num = np.array([float(i) for i in F_num], dtype=float)
    F_den = np.array([float(i) for i in F_den], dtype=float)
    aux_num = np.array([float(i) for i in aux_num], dtype=float)
    aux_den = np.array([float(i) for i in aux_den], dtype=float)
    return F_num, F_den, aux_num, aux_den
#%%

# -----------------------------
# STEP 2: Collect data (u, y)
# -----------------------------
# Here we simulate or assume you already have process data
N = 10000
u = np.random.randn(N)           # excitation input
model_time = N / Ts
y = np.zeros(N)

x=np.array([[0],[0]])

# SECOND order system, mass spring damper:
m=1
k=1
d=1
A=np.array([[0,1],[-k/m,-d/m]])
B=np.array([[0],[1/m]])
C=np.array([[1,0]])
D=0

c_system=signal.cont2discrete((A,B,C,D),Ts,method="zoh")
A_D=c_system[0]
B_D=c_system[1]
C_D=c_system[2]
D_D=c_system[3]

for k in range(0,N-1):
    s=A_D @ x[:,[k]] + B_D * u[k] + np.array([[0],[1]])*0.05*np.random.randn()
    x=np.append(x,s,1)
    y[k+1]= C_D @ x[:,[k]] + D_D * u[k]
# -----------------------------
# STEP 3: Construct virtual reference and error
# -----------------------------
# r_v = M^-1(z) y
# First invert M (apply filter defined by denominator/ numerator swapped)
# That is, r_v = lfilter(M_den, M_num, y)

M_num, M_den = M_cont_to_desc(tau,t,q,Ts)

lp_num, lp_den=W_cont_to_desc(omega,Ts)

F_num, F_den, aux_num, aux_den = GetFilterCoeff(M_num,M_den,lp_num,lp_den)
#r_v = lfilter(M_den, M_num, y) # THIS LINE DOES NOT WORK WITH TIME DELAYS!!

# Virtual, filtered error
e_v = lfilter(aux_num,aux_den,y) - lfilter(F_num,F_den,y)
# Filtered input
u_l=lfilter(F_num, F_den, u)
#%%
# -----------------------------
# STEP 4: Define controller structure C(q, θ)
# -----------------------------
#### PID regressor:
# the derivative term is implemented as the backwards difference.
# The integral terms is implemented as the cumulative sum of all error terms.
# C(z,θ): u= θ1 * e[k] + θ2 * (e[k]-e[k-1]) + θ3 * cumsum(e[k])
phi_PID=np.column_stack([e_v[1:],(e_v[1:]-e_v[:-1])/Ts,np.cumsum(e_v)[1:]*Ts])
#%%
# -----------------------------
# STEP 5: Solve least-squares problem
# -----------------------------
# We want C(q,θ) e_v ≈ u (the input that would produce desired closed-loop)

theta_PID, _, _, _ = lstsq(phi_PID, u_l[1:], rcond=None)
# -----------------------------
# STEP 6: Tuned controller
# -----------------------------
print("Tuned controller parameters (θ):", theta_PID)

# Assume theta from VRFT:

theta1_PID, theta2_PID, theta3_PID = theta_PID

# Clamp negative values for proportional and integral terms.
if theta1_PID<0:
    theta1_PID=0

if theta3_PID<0:
    theta3_PID=0
#%%
# -----------------------------
# STEP 7: Apply the tuned controller C(q, θ)
# -----------------------------


N_sim = 100
sim_time = [Ts*i for i in range(int(round(N_sim/Ts)))]

#various references are defined below.


r = np.ones(len(sim_time))   # step reference
y_cl = np.zeros(len(sim_time))
u_cl = np.zeros(len(sim_time))
e_cl = np.zeros(len(sim_time))
x_cl = np.array([[0],[0]]) 
for i in range(0,len(sim_time)):
    r[i]=np.sin(1.0*sim_time[i])

for i in range(0,len(sim_time)):
    if i<25/Ts:
        r[i]=1
    elif i>=25/Ts and i<50/Ts:
        r[i]=2
    elif i>=50/Ts and i<75/Ts:
        r[i]=3
    elif i>=75/Ts:
        r[i]=0

# Simulate system with active PID controller
for k in range(0,len(sim_time)-1):
    e_cl[k] = r[k] - y_cl[k]
    u_cl[k] = theta1_PID * e_cl[k] + theta2_PID * (e_cl[k]-e_cl[k-1]) + theta3_PID * (np.cumsum(e_cl)[k])*Ts
    
    s=A_D @ x_cl[:,[k]] + B_D*u_cl[k] #+ np.array([[0],[1]])*0.05*np.random.randn()
    x_cl = np.append(x_cl,s,1)
    y_cl[k+1] = C_D @ x_cl[:,[k]] + D_D*u_cl[k]

    
# -----------------------------
# STEP 8: Compare with reference model response
# -----------------------------
# Simulate desired closed-loop reference model M(q)
y_ref = lfilter(M_num, M_den, r, axis=-1)

# -----------------------------
# STEP 9: Plot results
# -----------------------------
plt.figure(figsize=(8,5))
plt.plot(sim_time,y_ref, 'g--', label="Reference model (desired)")
plt.plot(sim_time,y_cl, 'b', label="VRFT closed-loop output")
plt.plot(sim_time,r, 'k:', label="Reference input")
plt.plot(sim_time,u_cl,'r--',label="Actuator effort")
plt.xlabel("Time step")
plt.ylabel("Output")
plt.legend()
plt.title("VRFT Controller Performance vs Reference Model")
plt.grid(True)
plt.show()