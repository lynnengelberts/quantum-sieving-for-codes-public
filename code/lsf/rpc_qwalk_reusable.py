from random import uniform
from nns import *
from misc import *
from optimizer import *
from .rpc_qwalk import check_constraints_qwalk, wedge_size_LSF 

class RPC_QuantumWalk_Reusable(NNS):
    """ 
    Class that contains functions to calculate the runtime and the memory of RPC + Reusable Quantum Walk (with RPC) + Sparsification algorithm with optimized memory.
    Note that it only applies the reusable walk if the imposed condition is satisfied. 
    """

    _name = 'RPC_quantum_walk_reusable'    

    def check_constraint_reusable_walk(self, v: float, alpha: float, v_beta: float, beta: float):              
        """
        Returns 'True' iff the condition for applying the reusable walk from [BCSS23] is satisfied. 
        """
        num_sols_alpha_bucket = max(0, 2*self.bucket_size(v, alpha) + self.prob(v, alpha))  
        e_max = self.wedge_size(v, alpha)[0] # e^* that maximizes the wedge size 
        d_beta = comb(alpha, beta) + comb(v - alpha, v_beta - beta) - wedge_size_LSF(v, alpha, v_beta, beta, weight_overlap = e_max)[1]

        ncols = max(0, num_sols_alpha_bucket - d_beta) 
        size_codomain = comb(v, v_beta) - (comb(alpha, beta) + comb(v - alpha, v_beta - beta)) # 1/p_beta 
        if ncols <= size_codomain/4: 
            return True 
        else: 
            return False 

    def time_bucket_search(self, v: float, alpha: float, vertex_size: float, v_beta: float, beta: float): 
        # Quantities related to search in alpha-bucket 
        num_sols_alpha_bucket = max(0, 2*self.bucket_size(v, alpha) + self.prob(v, alpha))  

        # Quantities related to beta-bucketing 
        e_max = self.wedge_size(v, alpha)[0] # e^* that maximizes the wedge size 
        d_beta = comb(alpha, beta) + comb(v - alpha, v_beta - beta) - wedge_size_LSF(v, alpha, v_beta, beta, weight_overlap = e_max)[1]
        num_valid_beta_buckets = 0 # num_valid_beta_buckets 
        size_beta_bucket = vertex_size + comb(alpha, beta) + comb(v - alpha, v_beta - beta) - comb(v, v_beta) # Number of vertex elements that are in beta bucket 

        # Parameters quantum walk 
        delta = -vertex_size 
        epsilon = min(- d_beta, 2*vertex_size + self.prob(v, alpha) - d_beta) 
        setup = vertex_size + num_valid_beta_buckets 
        check = 0 
        update = max(num_valid_beta_buckets, (num_valid_beta_buckets + size_beta_bucket)/2)

        # Extra parameters reusable walk
        num_sols_per_beta_RPC = max(0, num_sols_alpha_bucket - d_beta)
        num_reps = num_sols_alpha_bucket - num_sols_per_beta_RPC

        # Apply reusable walk iff conditions are satisfied 
        if self.check_constraint_reusable_walk(v, alpha, v_beta, beta) == True:       
            return num_reps + max(setup, num_sols_per_beta_RPC + -epsilon/2 + max(update - delta/2, check))
        else: 
            return num_reps + num_sols_per_beta_RPC + max(setup, -epsilon/2 + max(update - delta/2, check))
    
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
            
        m_C = list_size(self._n, self._w)
        m_Q = vertex_size # num_valid_beta_buckets = 0 
        m_QRACM = self.bucket_size(v, alpha)
        m_QRAQM = m_Q
        return (m_C, m_Q, m_QRACM, m_QRAQM)
