import matplotlib.pyplot as plt
from misc import *
from nns import *

class GJN(NNS):
    """ 
    Class that contains functions to calculate the runtime and the memory of (classical) GJN algorithm.
    """

    _name = 'GJN'

    def runtime(self):
        # return list_size(1, w_NNS) + comb(w_NNS, w_NNS/2)
        return list_size(self._n, self._w) + comb(self._w, self._w/2)

    def memory(self):
        # return list_size(1, w_NNS) + comb(w_NNS, w_NNS/2)
        return list_size(self._n, self._w) + comb(self._w, self._w/2)