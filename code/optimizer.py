from abc import ABC, abstractmethod 
import scipy.optimize as opt

def validity(constrs, args, tol: float = 1e-7):
    """ 
    Used inside initialization function of optimizer to check validity of constraints for starting point 
    """
    list_of_constrs = [ (constr['type'], constr['fun'](args)) for constr in constrs ]
    for constr in list_of_constrs:
        if constr[0] == "eq":
            if abs(constr[1]) > tol:
                return False
        elif constr[1] < -tol:
            return False
    return True

class Optimizer(ABC):
    """ 
    Class that optimizes parameters (params) of optimization function (opt_func) in given number of iterations (iters) and for a given precision (prec).
    """

    @property
    @abstractmethod
    def opt_func(self):
        ...

    @property
    @abstractmethod
    def constrs(self):
        ...

    @property
    @abstractmethod
    def bounds(self):
        ...
    
    @property
    @abstractmethod
    def start(self):
        ...

    def optimize(self, iters: int = 100, prec: float = 1e-10, min_val: int = 1000, max_iter: int = 2000):
        """ 
        Optimizes parameters params of function opt_func in given number of iterations iter and for a given precision prec.
        """
        i = 0
        while i < iters:
            result = opt.minimize(self.opt_func, 
                                self.start, 
                                bounds = self.bounds,
                                constraints = self.constrs,
                                tol = prec, 
                                options = {'maxiter':max_iter})
            opt_val = result.get('fun')
            if (result.success and opt_val < min_val and validity(self.constrs, result.x, tol = prec)):
                min_val = opt_val 
                result_min = result 
            i += 1
        
        return result_min.x