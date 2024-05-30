from random import uniform
from nns import *
from misc import *
from optimizer import *

def check_constraints_qwalk(alg: NNS, v: float, alpha: float, vertex_size: float, v_beta: float, beta: float): 
    """ 
    Additional constraints for alpha-RPC instantiation.  
    Returns 'True' iff all constraints satisfied. 
    """
    # RPC constraints
    if v < alpha or alg._w < alpha or (alg._n - alg._w) < (v - alpha):  
        return False 

    # Additional constraints for QW version  
    if alg.bucket_size(v, alpha) < vertex_size: # Vertex size should be at most size of alpha-bucket 
        return False
    if v < v_beta or v_beta < beta or alpha < beta or (v - alpha) < (v_beta - beta):
        return False
    if 2*vertex_size > - alg.prob(v, alpha): # Because want 1/p >= s^2 
        return False
    
    # Check if bounds satisfied 
    if v > 1 or alpha < 0 or beta < 0 or vertex_size < 0: 
        return False 
    
    return True 

# The following could be merged with the existing wedge_size inside the NNS class
def wedge_size_LSF(n: float, w: float, v: float, alpha: float, weight_overlap: float = None, tol: float = 1e-10): 
        """
        Computes wedge quantities for the second layer of filtering. 
        """  
        if weight_overlap == None: 
            t = w/2 
        else:
            t = weight_overlap 
        def component_wedge_size(e):
            return comb(t, e) + 2*comb(w - t, alpha - e) + comb(n - 2*w + t, v - 2*alpha + e)
        def find_e(e):
            return -max(0, component_wedge_size(e))
        e=opt.fminbound(find_e, 0, min(t, alpha), xtol = tol, full_output = 1)
        return e[0], component_wedge_size(e[0])


class RPC_QuantumWalk(NNS):
    """ 
    Class that contains functions to calculate the runtime and the memory of RPC + Quantum Walk (with RPC) algorithm with optimized memory.
    """

    _name = 'RPC_quantum_walk'  
    
    def time_bucket_search(self, v: float, alpha: float, vertex_size: float, v_beta: float, beta: float): 
        # Quantities related to search in alpha-bucket 
        num_sols_alpha_bucket = max(0, 2*self.bucket_size(v, alpha) + self.prob(v, alpha))  

        # Quantities related to beta-bucketing 
        e_max = self.wedge_size(v, alpha)[0] # e^* that maximizes the wedge size 
        d_beta = comb(alpha, beta) + comb(v - alpha, v_beta - beta) - wedge_size_LSF(v, alpha, v_beta, beta, weight_overlap = e_max)[1] 
        num_valid_beta_buckets = d_beta
        size_beta_bucket = vertex_size + comb(alpha, beta) + comb(v - alpha, v_beta - beta) - comb(v, v_beta) # Number of vertex elements that are in beta bucket 

        # Parameters quantum walk 
        delta = -vertex_size 
        epsilon = min(0, 2*vertex_size + self.prob(v, alpha)) 
        setup = vertex_size + num_valid_beta_buckets 
        check = 0 
        update = max(num_valid_beta_buckets, (num_valid_beta_buckets + size_beta_bucket)/2)

        return num_sols_alpha_bucket + max(setup, -epsilon/2 + max(update - delta/2, check))

    def runtime(self, v: float, alpha: float, vertex_size: float, v_beta: float, beta: float):
        if isclose(v, alpha, abs_tol=1e-05): # Note that it currently doesn't update alpha in the main algorithm
            alpha = v 

        if check_constraints_qwalk(self, v, alpha, vertex_size, v_beta, beta) == False:
            return 100
        
        N = list_size(self._n, self._w) 
        P = comb(self._w, alpha) + comb(self._n - self._w, v - alpha)  
        D = self.wedge_size(v, alpha)[1]
        #F = comb(self._n, v) 
        R = P - D  
        num_buckets = self.num_buckets(v, alpha) # F - P 

        # Cost of bucketing phase and checking phase
        t_bucketing = N 
        t_checking = num_buckets + self.time_bucket_search(v, alpha, vertex_size, v_beta, beta)
        
        # Total cost of sieving
        t = R + max(t_bucketing, t_checking)
        return t
        
    def memory(self, v: float, alpha: float, vertex_size: float, v_beta: float, beta: float):
        if isclose(v, alpha, abs_tol=1e-05): # Note that it currently doesn't update alpha in the main algorithm
            alpha = v 
            
        # Quantities related to beta-bucketing 
        e_max = self.wedge_size(v, alpha)[0] # e^* that maximizes the wedge size 
        num_valid_beta_buckets = comb(alpha, beta) + comb(v - alpha, v_beta - beta) - wedge_size_LSF(v, alpha, v_beta, beta, weight_overlap = e_max)[1]

        m_C = list_size(self._n, self._w)
        m_Q = vertex_size + num_valid_beta_buckets
        m_QRACM = self.bucket_size(v, alpha)
        m_QRAQM = m_Q
        return (m_C, m_Q, m_QRACM, m_QRAQM)
    
    
class RPCOpt_QW(Optimizer):
    """ 
    Class that optimizes parameters of RPC + QW algorithm.
    """

    def __init__(self, alg: RPC_QuantumWalk):
        self._alg = alg

    @property
    def opt_func(self):
        return lambda args : self._alg.runtime(args[0], args[1], args[2], args[3], args[4])

    @property
    def bounds(self):
        return [(0, 1)] + [(0, self._alg._w)] + [(0,1)] + [(0,1)] + [(0,1)]

    @property
    def constrs(self):
        return [
            # For any RPC  
            { 'type' : 'ineq',   'fun' : lambda args_opt : args_opt[0] - args_opt[1]}, # v_ >= alpha
            { 'type' : 'ineq',   'fun' : lambda args_opt : self._alg._w - args_opt[1]}, # w_NNS >= alpha
            { 'type' : 'ineq',   'fun' : lambda args_opt : (1 - self._alg._w) - (args_opt[0] - args_opt[1])}, #(1 - w_NNS) >= (v - alpha), s.t. second binomial in CapVol is defined 
            # Additional for QW
            { 'type' : 'ineq',   'fun' : lambda args_opt : self._alg.bucket_size(args_opt[0], args_opt[1]) - args_opt[2]}, # bucket_size >= vertex_size   
            { 'type' : 'ineq',   'fun' : lambda args_opt : args_opt[0] - args_opt[3]}, # v_ >= v_beta
            { 'type' : 'ineq',   'fun' : lambda args_opt : args_opt[3] - args_opt[4]}, # v_beta >= beta
            { 'type' : 'ineq',   'fun' : lambda args_opt : args_opt[1] - args_opt[4]}, # alpha >= beta
            { 'type' : 'ineq',   'fun' : lambda args_opt : args_opt[0] - args_opt[1] - (args_opt[3] - args_opt[4])}, # v_ - alpha >= v_beta - beta
            { 'type' : 'ineq',   'fun' : lambda args_opt : - self._alg.prob(args_opt[0], args_opt[1]) - 2*args_opt[2]}, # s.t. 1/p >= s^2 
        ]
    
    @property
    def start(self, max_iter: int = 10000):
        i = 0
        while i < max_iter:
            v = uniform(0, 0.009)
            alpha = uniform(0, 0.009)
            vertex_size = uniform(0,0.009)
            v_beta = uniform(0,0.009)
            beta = uniform(0,0.009)
            start = [v, alpha, vertex_size, v_beta, beta]
            if validity(self.constrs, start): 
                return start
            i += 1
        return [100, 100]
