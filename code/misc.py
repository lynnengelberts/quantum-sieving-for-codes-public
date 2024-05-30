from math import *
import scipy.optimize as opt

def log2(x: float):
    """ 
    Logarithm base 2.
    """
    return log(x, 2)

def h(x: float): 
    """ 
    Binary entropy function.
    """ 
    if x < 0 or x > 1: # Return penalty if x not in [0,1]
        return -1000  
    
    if x == 0 or x == 1: 
        return 0.  
    return -x * log2(x) - (1 - x) * log2(1 - x)  

def comb(a: float, b: float):
    """
    Returns the log_2 of {a choose b} approximated by the binary entropy function. 
    """
    if(a <= 0.):
        return 0. 
    return a*h(b/a)

def h_inv(y: float):  
    """ 
    Inverse of the binary entropy function. 
    """ 
    if y == 1:
        return 0.5
    return opt.fsolve(lambda x: y - h(x), 0.0000001)[0]

def calc_w_from_GV(k: float): 
    """
    Returns weight corresponding to GV bound for rate k. 
    """
    return h_inv(1 - k)

def list_size(n: float, w: float):
        """
        Input list size s.t. number of NNS solutions (for target weight w_NNS) equals this list size.  
        """
        # return comb(n_NNS, w_NNS) - comb(w_NNS, w_NNS/2) - comb(n_NNS - w_NNS, w_NNS/2) 
        return comb(n, w) - comb(w, w/2) - comb(n - w, w/2)