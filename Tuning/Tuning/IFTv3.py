import numpy as np
from scipy.signal import lfilter
from scipy import signal
import matplotlib.pyplot as plt


import warnings 
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
# implement the simulation as a function:
# System parameters:
m, k ,d = 4, 1, 1

A=np.array([[0,1],[-k/m,-d/m]])
B=np.array([[0],[1/m]])
C=np.array([[1,0]])
D=0

c_system=signal.cont2discrete((A,B,C,D),Ts,method="zoh")
A_D, B_D,C_D,D_D = c_system[0], c_system[1], c_system[2], c_system[3]

def sys_sim(r,N,rho,w=True,Ts=Ts,A=A_D,B=B_D,C=C_D,D=D_D):
    sim_time = [Ts*i for i in range(int(round(N/Ts)))]
    K, D, I = rho[0], rho[1], rho[2]
    if w==True:
        v=1
    else:
        v=0
    y_cl = np.zeros(len(sim_time))
    u_cl = np.zeros(len(sim_time))
    e_cl = np.zeros(len(sim_time))
    x_cl = np.array([[0],[0]]) 
    for k in range(0,len(sim_time)-1):
        e_cl[k] = r[k] - y_cl[k]
        u_cl[k] = K * e_cl[k] + D * -(y_cl[k]-y_cl[k-1])/Ts + I * (np.cumsum(e_cl)[k])*Ts 
        
        s=A_D @ x_cl[:,[k]] + B_D*u_cl[k] + np.array([[0],[1]])*v*0.05*np.random.randn()
        x_cl = np.append(x_cl,s,1)
        y_cl[k+1] = C_D @ x_cl[:,[k]] + D_D*u_cl[k]

    return y_cl, u_cl, e_cl
#%%
# Simulation and discretisation parameters:
Ts=0.1
N=100 # Simulation time [s]
sim_time = [Ts*i for i in range(int(round(N/Ts)))] #list of all time steps.

#%%
r = np.ones(len(sim_time))
for i in range(0,len(sim_time)):
    if i<25/Ts:
        r[i]=1
    elif i>=25/Ts and i<50/Ts:
        r[i]=2
    elif i>=50/Ts and i<75/Ts:
        r[i]=3
    elif i>=75/Ts:
        r[i]=0
        
rho = [0.20187259, 0.43799839, 0.05353617]
M_num, M_den = M_cont_to_desc(tau,t,q,Ts)
y_ref = lfilter(M_num, M_den, r, axis=-1)

def IFT(rho,r,it=5,lam=0):
    rho_vec=[]
    for k in range(it+1):
        y_ref = lfilter(M_num, M_den, r, axis=-1)

        y_1,u_1,e_1 = sys_sim(r,N,rho)

        y_2,u_2,e_2 = sys_sim((r-y_1),N,rho)

        y_3,u_3,e_3 = sys_sim(r,N,rho)

        y_tilde = y_1- y_ref
        # Stuff for finding dy/drho
        #following is from equation 19 in IFT theory and applications paper
        #filter y_2 through dCy/dRho:
        third1 = lfilter([1],[1],y_2)
        third2 = lfilter([1,-1],[Ts],y_2)
        third3 = lfilter([Ts],[1,-1],y_2)
        #filter y_3 through dCr/dRho-dCy/dRho:
        second1 = lfilter([0],[1],y_3)
        second2 = lfilter([-1,1],[Ts],y_3)
        second3 = lfilter([0],[1],y_3)
        #sum up the previous expressions:
        comb1 = second1+third1
        comb2 = second2+third2
        comb3 = second3+third3
        #filter the sum though (Cr(Rho))^-1
        dydrho1 = lfilter([1,-1],[(rho[0]+rho[2]*Ts),-rho[0]],comb1)
        dydrho2 = lfilter([1,-1],[(rho[0]+rho[2]*Ts),-rho[0]],comb2)
        dydrho3 = lfilter([1,-1],[(rho[0]+rho[2]*Ts),-rho[0]],comb3)
        
        #tuff for finding du/drho
        #following is from equation 27
        third1 = lfilter([1],[1],u_2)
        third2 = lfilter([1,-1],[Ts],u_2)
        third3 = lfilter([Ts],[1,-1],u_2)
        
        second1 = lfilter([0],[1],u_3)
        second2 = lfilter([-1,1],[Ts],u_3)
        second3 = lfilter([0],[1],u_3)
        
        comb1 = second1+third1
        comb2 = second2+third2
        comb3 = second3+third3
        
        dudrho1 = lfilter([1,-1],[(rho[0]+rho[2]*Ts),-rho[0]],comb1)
        dudrho2 = lfilter([1,-1],[(rho[0]+rho[2]*Ts),-rho[0]],comb2)
        dudrho3 = lfilter([1,-1],[(rho[0]+rho[2]*Ts),-rho[0]],comb3)
        #following is from equation 28
        djdrho1=0
        djdrho2=0
        djdrho3=0
        
        for i in range(len(sim_time)-1):
            djdrho1=djdrho1+y_tilde[i]*dydrho1[i]+lam*u_1[i]*dudrho1[i]
            djdrho2=djdrho2+y_tilde[i]*dydrho2[i]+lam*u_2[i]*dudrho2[i]
            djdrho3=djdrho3+y_tilde[i]*dydrho3[i]+lam*u_3[i]*dudrho3[i]
            
        djdrho1=djdrho1/(len(sim_time)-1)
        djdrho2=djdrho2/(len(sim_time)-1)
        djdrho3=djdrho3/(len(sim_time)-1)

        djdrho=np.array([djdrho1,djdrho2,djdrho3])
        #The following is from equation 33
        dydrho=np.array([dydrho1,dydrho2,dydrho3])
        dudrho=np.array([dudrho1,dudrho2,dudrho3])
        
        Tdydro=dydrho.transpose()
        Tdudro=dudrho.transpose()
        
        R=np.array([[0,0,0],[0,0,0],[0,0,0]])
        for i in range(len(sim_time)-1):
            R=R+dydrho[:,[i]]@Tdydro[[i],:] + lam*dudrho[:,[i]]@Tdudro[[i],:]
            
        R=R/(len(sim_time)-1)
        Rinv=np.linalg.inv(R)
        rho0=np.array([rho[0],rho[1],rho[2]])

        rho=rho0-Rinv@djdrho
        print("iteration",k,"parameters:",rho)
        rho_vec.append(rho)
    return rho, rho_vec


#%%
it = 20

rho,rho_vec = IFT(rho,r,it,0)


#%%
# sim stuff


N_sim = 100
sim_time = [Ts*i for i in range(int(round(N_sim/Ts)))]

#various references are defined below.


r = np.ones(len(sim_time))   # step reference
y_cl = np.zeros(len(sim_time))
u_cl = np.zeros(len(sim_time))
e_cl = np.zeros(len(sim_time))
x_cl = np.array([[0],[0]]) 

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

y_cl,_,_=sys_sim(r, N, rho,w=False)
y_cl_1,_,_=sys_sim(r, N, rho_vec[1],w=False)
y_cl_5,_,_=sys_sim(r, N, rho_vec[5],w=False)
y_cl_8,_,_=sys_sim(r, N, rho_vec[8],w=False)
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
plt.plot(sim_time,y_cl, 'b', label="IFT closed-loop output,it "+str(it))

plt.plot(sim_time,r, 'k:', label="Reference input")
#plt.plot(sim_time,u_cl,'r--',label="Actuator effort")
plt.xlabel("Time step")
plt.ylabel("Output")
plt.legend()
plt.title("Controller Performance vs Reference Model. Num iterations = "+str(it))
plt.grid(True)
plt.show()
