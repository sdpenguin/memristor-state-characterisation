import numpy as np

Vt = 0.026 # Thermal voltage

# Our model
def get_f_proposed(a1, a2, b1, b2, Gp, s1, s2, s3):
    def f(v, G):
        return G*(a1*(np.exp(b1*v) - 1) + a2*(1 - np.exp(-b2*v)) + Gp*v)
    return f

# MSS
def get_f_mss(a1, a2, b1, b2, Gp, s1, s2, s3):
    def f(v, G):
        return a1*np.exp(b1*v) - a2*np.exp(-b2*v) + G*Gp*v
    return f

# Modified MSS
def get_f_mss_mod(a1, a2, b1, b2, Gp, s1, s2, s3):
    def f(v, G):
        return a1*(np.exp(b1*v) - 1) + a2*(1 - np.exp(-b2*v)) + G*Gp*v
    return f
