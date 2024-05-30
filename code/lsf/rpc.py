from random import uniform
from nns import *
from misc import *
from optimizer import *

def check_constraints(alg: NNS, v: float, alpha: float): 
    """ 
    Additional constraints for alpha-RPC instantiation.  
    Returns 'True' iff all constraints satisfied. 
    """
    if v < alpha or alg._w < alpha or (alg._n - alg._w) < (v - alpha):  
        return False 
    
    # Check if bounds satisfied 
    if v > 1 or alpha < 0: 
        return False 
    
    return True 


class RPC(NNS):
    """ 
    Class that contains functions to calculate the runtime and the memory of RPC algorithm with optimized memory.
    """

    _name = 'RPC'

    def runtime(self, v: float, alpha: float):
        if check_constraints(self, v, alpha) == False:
            return 100
        
        N = list_size(self._n, self._w)
        P = comb(self._w, alpha) + comb(self._n - self._w, v - alpha)  
        D = self.wedge_size(v, alpha)[1]
        F = comb(self._n, v) 

        t = N + P - D + max(0, N + P - F)
        return t

    def memory(self, v: float, alpha: float):
        return list_size(self._n, self._w)
    
    
class RPCOpt(Optimizer):
    """ 
    Class that optimizes parameters of RPC algorithms.
    """

    def __init__(self, alg: RPC):
        self._alg = alg

    @property
    def opt_func(self):
        return lambda args : self._alg.runtime(args[0], args[1])

    @property
    def bounds(self):
        # bounds_RPC = [(0, 1)] + [(0, w_NNS)] 
        return [(0, 1)] + [(0, self._alg._w)]

    @property
    def constrs(self):
        return [
            { 'type' : 'ineq',   'fun' : lambda args_opt : args_opt[0] - args_opt[1]}, # v_ >= alpha
            { 'type' : 'ineq',   'fun' : lambda args_opt : self._alg._w - args_opt[1]}, # w_NNS >= alpha
            { 'type' : 'ineq',   'fun' : lambda args_opt : (1 - self._alg._w) - (args_opt[0] - args_opt[1])}, #(1 - w_NNS) >= (v - alpha), s.t. second binomial in CapVol is defined 
        ]
    
    @property
    def start(self, max_iter: int = 10000):
        i = 0
        while i < max_iter:
            v = uniform(0, 0.009)
            alpha = uniform(0, 0.009)
            start = [v, alpha]
            if validity(self.constrs, start): 
                return start
            i += 1
        return [100, 100]