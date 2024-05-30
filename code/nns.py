from abc import ABC, abstractmethod 
import matplotlib.pyplot as plt
import os
import scipy.optimize as opt
from misc import *

class NNS(ABC):
    """ 
    Class that contains functions to calculate the runtime and the memory of (classical) NNS algorithms.
    """

    _name = 'NNS'

    def __init__(self, n: float = 1, w: float = 0.5):
        self._n = n
        self._w = w

    def num_buckets(self, v: float, alpha: float): 
        """ 
        Log_2 of the expected number of buckets. 
        """  
        return comb(self._n, v) - comb(self._w, alpha) - comb(self._n - self._w, v - alpha)

    def bucket_size(self, v: float, alpha: float): 
        """ 
        Log_2 of the expected number of elements from L with |L| = output_size(n, w) in an alpha-bucket of a center point in S_v^n. 
        """ 
        return list_size(self._n, self._w) + comb(v, alpha) + comb(self._n - v, self._w - alpha) - comb(self._n, self._w)

    def wedge_size(self, v: float, alpha: float, weight_overlap: float = None, tol: float = 1e-10): 
        """
        Returns e and the log_2 size of the wedge in S_v^n defined by vectors x,y of weight w such that |x \land y| = weight_overlap, where e is the dominating contributor to the wedge size.
        """  
        if weight_overlap == None: 
            t = self._w/2 
        else:
            t = weight_overlap 
        def component_wedge_size(e):
            return comb(t, e) + 2*comb(self._w - t, alpha - e) + comb(self._n - 2*self._w + t, v - 2*alpha + e)
        def find_e(e):
            return -max(0, component_wedge_size(e))
        e=opt.fminbound(find_e, 0, min(t, alpha), xtol = tol, full_output = 1)
        return e[0], component_wedge_size(e[0])

    def prob(self, v: float, alpha: float):   
        """
        Returns the (log_2 of the) probability that a pair of alpha-bucket elements forms a solution, i.e. Pr_{x,y in Bucket_alpha(c)}[|x+y|=w] for c of weight v and x,y of weight w_NNS, each being a vector of dimension n_NNS.
        NB: This probability is independent of the NNS-instantiation (e.g., it holds for both Theorem 4.4 and Corollary 4.2 in [DEEK23]). See Overleaf for a derivation. 
        """
        return comb(self._w, self._w/2) + comb(self._n - self._w, self._w/2) - comb(v, alpha) - comb(self._n - v, self._w - alpha) - comb(self._w, alpha) - comb(self._n - self._w, v - alpha) + self.wedge_size(v, alpha)[1] 

    @abstractmethod
    def runtime(self):
        ...

    @abstractmethod
    def memory(self):
        ...


def plot_times(collection_of_results: list, directory: str = 'plots/', name: str = 'NNS_times'):
    plt.clf()
    plt.style.use('tableau-colorblind10')
    for result in collection_of_results:
        L_opt = result[0]
        label_result = result[1]
        L_plot =[[j[0],j[1]] for j in L_opt] 
        x,y=zip(*L_plot)
        plt.plot(x, y, label=label_result)  
    plt.title("Time complexity of code sieving for varying weight")
    plt.xlabel(r'$\omega$' + ' s.t. weight is ' + r'$w = \omega n$')
    plt.ylabel(r'$c$' + ' s.t. runtime is ' + r'$2^{cn}$')
    plt.legend(bbox_to_anchor=(0.61, 1), loc='upper left')

    if not os.path.exists(directory):
	    os.makedirs(directory)
    #plt.savefig(directory + name + '.eps', format='eps') 
    plt.savefig(directory + name + '.png')


def plot_memory(collection_of_results: list, directory: str = 'plots/', name: str = 'NNS_memory'): 
    plt.clf()
    for result in collection_of_results:
        L_opt = result[0] 
        label_result = result[1]
        L_plot =[[j[0],j[2]] for j in L_opt] 
        x,y=zip(*L_plot)
        plt.plot(x, y, label=label_result)  
    plt.title("Memory complexity of code sieving for varying weight")
    plt.xlabel(r'$\omega$' + ' s.t. weight is ' + r'$w = \omega n$')
    plt.ylabel(r'$c$' + ' s.t. runtime is ' + r'$2^{cn}$')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    if not os.path.exists(directory):
	    os.makedirs(directory)
    plt.savefig(directory + name + '.png')
