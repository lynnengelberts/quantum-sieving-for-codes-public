import os
import csv
from nns import *
from lsf.gjn import *
from lsf.rpc import *
from lsf.rpc_grover import *
from lsf.rpc_qwalk import * 
from lsf.rpc_qwalk_spars import * 
from lsf.rpc_qwalk_reusable import *

def alg_choice(alg_name):
    match alg_name:
        case 'GJN':
            alg = GJN()
            optimizer = None
            label = 'GJN'
        case 'RPC':
            alg = RPC()
            optimizer = RPCOpt(alg)
            label = 'Classical'
        case 'RPC_Grover':
            alg = RPC_Grover()
            optimizer = RPCOpt(alg) # Same as classical 
            label = 'Grover'
        case 'RPC_quantum_walk': 
            alg = RPC_QuantumWalk()
            optimizer = RPCOpt_QW(alg)
            label = 'QW + LSF'
        case 'RPC_quantum_walk_sparsification': 
            alg = RPC_QuantumWalk_Sparsification()
            optimizer = RPCOpt_QW(alg) # Same as QW without sparsification 
            label = 'QW + LSF + Spars.'
        case 'RPC_quantum_walk_reusable': 
            alg = RPC_QuantumWalk_Reusable()
            optimizer = RPCOpt_QW(alg) # Same as QW without sparsification 
            label = 'Reus. QW + LSF + Spars.'
        case _:
            print('This algorithm is not available.')
    return alg, optimizer, label 

##########################################################################
#---------------------- HELPER FUNCTIONS --------------------------------#
##########################################################################

def write_results(res: str, dir: str, filename: str):
    """ 
    Write results res into dir/filname.csv.
    """
    if not os.path.exists(dir):
        os.makedirs(dir)
    with open(dir + filename + '.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=' ')
            writer.writerows(res)

def read_results(dir: str, filename: str):
    """ 
    Reads results res into dir/filname.csv.
    """
    res = []
    with open(dir + filename, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=' ')
        for row in reader:
            row_float = [float(r) for r in row]
            res.append(row_float)
    return res

def time_memory(alg, optimizer = None, range_weights = 100, iters = 50, prec = 1e-7, min_val = 1000, max_iter = 2000): 
    """
    Construct list containing all [w, t] for different w, where t is the optimum time found in given iterations for given precision. 
    Writing r=range_weights, w ranges over [1/r, 1/2) in steps of 1/r. 
    """

    res = []
    for i in range(1, int(range_weights/2)):  
        print("i: ", i)
        w = i/range_weights
        if w >= 0.49: # LE: Only until <0.49, because QW version seems to have issues with 49. TODO: Resolve 
            continue 
        alg._n = 1
        alg._w = w
        if alg._name == 'GJN':
            t = alg.runtime()
            m = alg.memory()
        else:
            v, alpha, *args = optimizer.optimize(iters, prec, min_val, max_iter)
            t = alg.runtime(v, alpha, *args)
            m = alg.memory(v, alpha, *args)
        res.append([w, t, m, [v, alpha, *args]])
 
        if alg._name == 'RPC_quantum_walk_reusable': 
            v, alpha, s, v_beta, beta = v, alpha, *args
            if alg.check_constraint_reusable_walk(v, alpha, v_beta, beta) == True: # Check if reusable is applied
                print("Reusable walk applied")
    return res
    
##########################################################################
#--------------------------- DRIVER CODE --------------------------------#
##########################################################################
plots_dir = '../plots/'
data_dir = '../data/'

range_weights = 100
iters = 20
prec = 1e-7

# Worst-case complexity
max_t_in_L = lambda L : max(L, key=lambda x: x[1])[1]

# Algorithms
alg_names = ['RPC', 'RPC_Grover', 'RPC_quantum_walk', 'RPC_quantum_walk_sparsification', 'RPC_quantum_walk_reusable']

result = []
for alg_name in alg_names:
    alg, optimizer, alg_label = alg_choice(alg_name)
    L = time_memory(alg, optimizer, range_weights, iters, prec)
    write_results(L, data_dir, alg_name + '_w' + str(range_weights) + '_i' + str(iters) + '_p' + str(prec))
    print(alg_name, "worst-case complexity :", max_t_in_L(L))
    result.append([L, alg_label])

# Plot
plot_times(result, plots_dir, 'NNS' + '_w' + str(range_weights) + '_i' + str(iters) + '_p' + str(prec))

##########################################################################
##########################################################################
print('----------------------------')