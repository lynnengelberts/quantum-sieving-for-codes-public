from math import *
import scipy.optimize as opt
import matplotlib.pyplot as plt
from misc import comb, calc_w_from_GV, list_size


def p2(k_ISD, n_NNS, w_NNS):   
    """ 
    Return p2, one of the components of the success probability of SievingISD.
    """
    return list_size(n_NNS, w_NNS) + (n_NNS - k_ISD) - comb(n_NNS, w_NNS)

def check_constraints(k_ISD, n_NNS, w_NNS): 
    """ 
    Returns true iff general constraints for SievingISD are satisfied.
    """ 
    w_ISD = calc_w_from_GV(k_ISD)
    if 1 < n_NNS or w_ISD < w_NNS: 
        return False 
    if n_NNS < w_NNS or (1 - n_NNS) < (w_ISD - w_NNS):  
        return False 
    if comb(n_NNS, w_NNS) < (n_NNS - k_ISD): # NB: Included in p2 <= 1, but considered separately as it is not sufficient for the lower bound claim 
        return False     
    if p2(k_ISD, n_NNS, w_NNS) > 0: # NOTE: Without this constraint, quantum Prange isn't better.
        return False 
    return True


### Functions that are to be compared: lower bound on Quantum SievingISD and quantum Prange 

def quantum_SievingISD_lower_bound(k_ISD, n_NNS, w_NNS):
    """
    Returns lower bound (naive) on the runtime of Quantum SievingISD, where is assumed that the oracle has runtime N.
    """ 
    w_ISD = calc_w_from_GV(k_ISD)
    
    # Uncomment if want to put a lower bound on n_NNS - k_ISD 
    # if isclose(n_NNS, k_ISD, abs_tol=0.0001): 
    #     return 1000
    
    if check_constraints(k_ISD, n_NNS, w_NNS) == False: 
        return 1000
    N_min = list_size(n_NNS, w_NNS)
    runtime_lower_bound = (N_min + comb(1, w_ISD) - (n_NNS - k_ISD) - comb(1 - n_NNS, w_ISD - w_NNS))/2
    return runtime_lower_bound    

def quantum_Prange(k_ISD):
    """ 
    Returns runtime of quantum Prange.
    """
    w_ISD = calc_w_from_GV(k_ISD)
    return (comb(1, w_ISD) - comb(1 - k_ISD, w_ISD))/2


### Functions for optimizing quantum_SievingISD_lower_bound 

def lower_bound_optimize_w(k_ISD, n_NNS, prec = 1e-7): 
    """ 
    Finds optimal w_NNS (and corresponding lower bound) for given n_NNS.
    """ 
    def lower_bound_for_fixed_n(w): 
        return quantum_SievingISD_lower_bound(k_ISD, n_NNS, w)
    w_ISD = calc_w_from_GV(k_ISD)
    w_opt = opt.fminbound(lower_bound_for_fixed_n, 0, min(w_ISD, n_NNS), xtol=prec, full_output=1)
    w_NNS = w_opt[0]
    
    return w_NNS, quantum_SievingISD_lower_bound(k_ISD, n_NNS, w_NNS)

def optimize_lower_bound(k_ISD, prec = 1e-7): 
    """ 
    Finds optimal n_NNS and w_NNS (and corresponding lower bound). 
    """
    def lower_bound(n): 
        w_NNS = lower_bound_optimize_w(k_ISD, n, prec)[0]
        return quantum_SievingISD_lower_bound(k_ISD, n, w_NNS)
    n_opt = opt.fminbound(lower_bound, k_ISD, 1, xtol=prec, full_output=1)
    n_NNS = n_opt[0]
    w_NNS = lower_bound_optimize_w(k_ISD, n_NNS, prec)[0]
    
    print("k_ISD, n_NNS, w_NNS = ", k_ISD, n_NNS, w_NNS)
    return n_NNS, w_NNS, quantum_SievingISD_lower_bound(k_ISD, n_NNS, w_NNS)


### Functions for checking our claim 

def check_conjecture_using_fminbound(range_rates = 100, prec = 1e-7):
    lst_of_lb = [] 
    lst_of_qP = [] 
    lst_of_checks = [] 

    num_of_violations = 0 # Check whether conjecture seems reasonable 

    for i in range(1, range_rates):
        k_ISD = i/range_rates
        # if k_ISD > 0.96:
        #     continue
        time_lower_bound = optimize_lower_bound(k_ISD, prec)[2]
        time_quantum_Prange = quantum_Prange(k_ISD)
        lst_of_lb.append([k_ISD, time_lower_bound])
        lst_of_qP.append([k_ISD, time_quantum_Prange])

        # Check if current iteration satisfies conjecture 
        if time_lower_bound > time_quantum_Prange: 
            check = 1
        else: 
            check = 0
            num_of_violations += 1
            print("Conjecture violated")
        lst_of_checks.append(check)

    print("Number of violations = ", num_of_violations, " out of ", range_rates) 

    lst_of_times = [lst_of_lb, lst_of_qP]
    return lst_of_times, lst_of_checks 

def plot_comparison(lst_of_times): #lst_of_times contains lb and qP times 
    L_lb = lst_of_times[0]
    label_lb = "lower bound Quantum SievingISD"
    L_qP = lst_of_times[1]
    label_qP = "quantum Prange"
    for version in [[L_lb, label_lb], [L_qP, label_qP]]:
            k,t = zip(*version[0])
            plt.plot(k, t, label=version[1])  
    plt.title("Lower bound on Quantum SievingISD versus quantum Prange")
    plt.xlabel(r'$\kappa$' + " s.t. k = " + r'$\kappa$' + "n")
    plt.ylabel(r'$c$' + ' s.t. (lower bound on) runtime is ' + r'$2^{cn}$') 
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.show()


### Function to plot the lower bound as a function of n',w' (for given k)

def plot_lower_bound(k_ISD, range_of_n_NNS = 100, range_of_w_NNS = 500):
    results = []
    for i in range(range_of_n_NNS): 
        L_lb = []
        n_NNS = k_ISD + i/range_of_n_NNS 
        if n_NNS >= 1: 
            continue

        for i in range(range_of_w_NNS):
            w_NNS = i/range_of_w_NNS
            if check_constraints(k_ISD, n_NNS, w_NNS) == False: 
                continue
            
            lower_bound = quantum_SievingISD_lower_bound(k_ISD, n_NNS, w_NNS)
            L_lb.append([k_ISD, n_NNS, w_NNS, lower_bound]) # lower bound on quantum ISD

        if L_lb != []: 
            results.append([L_lb, "n' = " + str(n_NNS)])

    for result in results:
        L = result[0] 
        label = result[1]
        L_plot =[[j[2],j[3]] for j in L] # Only plot w_NNS and lower_bound
        x,y=zip(*L_plot)
        plt.plot(x, y, label=label)  
    plt.title("Lower bound on Quantum SievingISD as function of (n', w') for k = " + str(k_ISD))
    plt.xlabel(r'$\omega$' + " s.t. w' = " + r'$\omega$' + "n'")
    plt.ylabel(r'$c$' + ' s.t. runtime is ' + r'$2^{cn}$')  
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.show()


### Driver code
    
lst_of_times, lst_of_checks = check_conjecture_using_fminbound(range_rates = 100, prec = 1e-10) 
plot_comparison(lst_of_times)

k_ISD = 0.44
plot_lower_bound(k_ISD, range_of_n_NNS = 100, range_of_w_NNS = 500)
