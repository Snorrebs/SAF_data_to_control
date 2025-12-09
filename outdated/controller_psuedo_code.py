import numpy as np
from scipy import signal
from scipy.signal import lfilter
#Import...


def control(r,rho,omega = 4,Ts = 1):
    # r is a reference signal,
    # rho is a vector containing the controller parameters,
    # Ts is the discretisation interval, assumed to be 1 second.
    # omega is the time constant imposed on the electrode dynamics
    
    N = len(r) # sim time in seconds
->  hist = history()... # initialise plant history. Requres ten(?) data points from wacker dataset

    y = np.zeros(N) # initialise resistance output
    e = np.zeros(N) # initialise error 
    u = np.zeros(N) # initialise inputs (electrode height)
    
->  y[0] = hist.resistance()... # initialise first entry in y as the last resistance in history
    
->  u[0] = hist.holder_pos()... # initialise holder positions
->  u_f = hist.holder_pos_all()... # initialise with all holder position in history. This is used to implement electrode dynamics.
    
    # the following generates a lowpass filter used to artificially impose dynamics on the elctrodes.
    W_CT = signal.TransferFunction([omega], [1, omega])
    W_DT = W_CT.to_discrete(Ts, method='bilinear')
    W_num, W_den = W_dt.num, W_dt.den
    
    for i in range(0,N-1):
        # implement basic PID controller. Derivative acts on state, not input.
        e[i] = r[i] - y[i]
        u_f[i+10] = rho[0] * e[i] - rho[1]*(y[i] - y[i-1])/Ts + rho[2]*np.cumsum(e[i])*Ts
        
        u[i] = lfilter([W_num],[W_den],u_f)[i+10] # low passes inputs to prevent teleportation
        
->      current_data = Simulator(hist,u[i],other_static_data,sim_time = 1...) # input history, lowpassed elctrode height, and potentially other data 
        
->      fused_data = data_fusion(curent_data,...) # possibly implement data fusion. Assume that the fused data contains the same states as current_data
        
        
->      hist = hist.append(fused_data) # Append output of simulator 
->      hist = hist[1:10] # move history window 1 step by dropping first entry.
        
->      y[i+1] = fused_data.resistance() # Pick out resistance from dataset