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

class RPC_Grover(NNS):
    """ 
    Class that contains functions to calculate the runtime and the memory of RPC + Grover algorithm with optimized memory.
    """

    _name = 'RPC_Grover'

    def time_bucket_search(self, v: float, alpha: float): 
        size_bucket = self.bucket_size(v, alpha)  
        return max(0, size_bucket, 2*size_bucket + self.prob(v, alpha)/2) 
    
    def runtime(self, v: float, alpha: float):
        if check_constraints(self, v, alpha) == False:
            return 100     
        
        N = list_size(self._n, self._w) 
        P = comb(self._w, alpha) + comb(self._n - self._w, v - alpha)  
        D = self.wedge_size(v, alpha)[1]
        #F = comb(self._n, v) 
        R = P - D  
        num_buckets = self.num_buckets(v, alpha) # F - P 

        # Cost of bucketing phase and checking phase
        t_bucketing = N 
        t_checking = num_buckets + self.time_bucket_search(v, alpha)

        # Total cost of sieving
        t = R + max(t_bucketing, t_checking)
        return t

    def memory(self, v: float, alpha: float):
        m_C = list_size(self._n, self._w)
        m_Q = 0
        m_QRACM = self.bucket_size(v, alpha)
        m_QRAQM = 0
        return (m_C, m_Q, m_QRACM, m_QRAQM)
