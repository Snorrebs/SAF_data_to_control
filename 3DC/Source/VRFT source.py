import numpy as np
import sympy as sm
from scipy import signal

# Specify a reference closed loop transfer function on the form:
# M(z)=e^(-tau*s)/(1+0.2*t*s)^q
# Where tau is a time delay, t is the settling time for the poles, and q is the system order. 
# M_cont_to_disc takes as inputs q,t,tau and the discretisation intervall.
# It returns the numerator and denominator of the discretised transfer function.

def M_cont_to_disc(tau,t,q,Ts):
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

# Specify a frequency weighting function on the form:
# W(z) = omega/(omega+s)
# where omega is the cut-off frequency.
# W_cont_to_disc takes as inputs omega and the discretisation interval.
# It returns the numerator and denominator of the discretised frequency weighting filter.
def W_cont_to_disc(omega,Ts):
    W_CT = signal.TransferFunction([omega], [1, omega])
    W_DT = W_CT.to_discrete(Ts, method='bilinear')
    return W_DT.num, W_DT.den



# GetFilterCoeff takes as input arguments the numerator and denominator of M(z), W(z) and Phi^(-1/2)(z). It returns the numerators and denominators of the filters:
# F = M * (1 - M) * W * Phi^(-1/2)
# F_aux = (1 - M) * W * Phi^(-1/2)
# NOTE: Phi is a scalar function that has the same frequency response as the spectral power of the inputs u[t].
# This function must be fitted manually, and must be a rational polynomial.
# It is assumed that the numerator and denominator of Phi^(-1/2) are provided directly to GetFilterCoeff. 
def GetFilterCoeff(num,den,lp_num,lp_den,phi_num,phi_den):
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
    W = subConstructRational(lp_num,lp_den) # Construct frequency weighting function.
    Phi_Sqrt_inv = subConstructRational(phi_num,phi_den) # construct Phi^(-1/2)
 
    F = M * (1 - M) * W * Phi_Sqrt_inv  # construct filter used on e_vr(t) and u(t)
    F_num, F_den = subGetCoeffs(F) #Get coefficients of the filter
    F_aux = (1 - M) * W * Phi_Sqrt_inv  #auxiliary filter to avoid having to calculate r_v directly, thus circumventing the scipy bug.
    aux_num, aux_den = subGetCoeffs(F_aux)
      
    #convert from sympy float to numpy float
    
    F_num = np.array([float(i) for i in F_num], dtype=float)
    F_den = np.array([float(i) for i in F_den], dtype=float)
    aux_num = np.array([float(i) for i in aux_num], dtype=float)
    aux_den = np.array([float(i) for i in aux_den], dtype=float)
    return F_num, F_den, aux_num, aux_den






