import torch 
import matplotlib.pyplot as plt
import numpy as np
import itertools
import os
import scipy.io
from scipy.sparse import random
from scipy.optimize import linear_sum_assignment
from softmax_optimization_sparse import lap_softmax
import time

n = 1000
m = 20
maxIter = 10000
gamma = 20.0
learning_rate = 1e-3
# "multi" decides if to select optimizing only for the nonzero entries (multi=0.0), otherwise multi = 1.0 (or greater if you wish):
multi = 1

output_folder = f'result_{multi}'
faust_folder = './data/'

use_GPU = True
torch.manual_seed(0)

subsample = np.arange(n)
if not os.path.exists(output_folder):
    os.mkdir(output_folder)
    
def create_sparse_A(sz=5, density = 1.0):
    sparse_matrix = random(sz, sz, random_state=42, density=density, format = 'lil')
    sparse_matrix = scipy.sparse.triu(sparse_matrix,format = 'lil')
    sparse_matrix_l = scipy.sparse.triu(sparse_matrix,k=1,format = 'lil').T
    sym =  (sparse_matrix+sparse_matrix_l)
    sym = -sym
    rows, cols = sym.nonzero()
    A_max_nonzero = sym[cols, rows].max()

    if abs(A_max_nonzero) > 1:
        sym = sym/A_max_nonzero
    return sym

perms = []
rel_errors = []

thresh = lambda x: (x > 0.5)
def argmaxing(x):
    x_argmax = np.zeros(x.shape)
    argmax_inds = x.argmax(axis=1)
    x_argmax[np.arange(x.shape[0]),argmax_inds] = 1
    return x_argmax

def make_perm(Pin):
    P = Pin.cpu().detach().clone()
    P_new = torch.zeros_like(P)
    for i in range(P.shape[0]):
        mx, my = np.unravel_index(np.argmax(P, axis=None), P.shape)
        P_new[mx, my] = 1
        P[mx,:]=-1e5
        P[:,my]=-1e5
    return P_new


multis = [1000,5000] # Choose Problem Size
for i in range(0,len(multis)):
    
    n=multis[i]
    A = create_sparse_A(n,0.01)
    if abs(A.max()) > 1:
        A = A/A.max()
    idc_nz = A.nonzero()
    idc = (list(idc_nz[0]),list(idc_nz[1]))
    A_t = torch.tensor(A.data).cuda()
    costs = lambda x: torch.sum(torch.tensor(A.todense()) * x)

    # LAP
    solution = linear_sum_assignment(torch.tensor(A.todense()))
    
    # costs ground truth
    x_gt = np.zeros([n, n])
    for j in solution[0]:
        x_gt[j, solution[1][j]] = 1
    costs_gt = costs(x_gt).numpy()
    print(costs_gt)
        
    # optimization
    t = time.time()
    
    x_solution, V, W,grad_save = lap_softmax(A_t, gamma, maxIter, learning_rate, n, m, use_GPU, idc, multi=multi) 
    elapsed = time.time() - t
    
    # evaluation
    costs_gt = costs(x_gt).numpy()
    
    # argmaxing
    x_thresh = argmaxing(x_solution)

    # is permutation
    is_perm = False
    if (x_thresh@np.ones([n, 1]) == 1).all() and (np.ones([1,n])@x_thresh == 1).all():
        is_perm = True
        
    # costs
    costs_solution = costs(x_solution).numpy()
    print(costs_solution)
    costs_solution_perm = costs(make_perm(torch.tensor(x_solution))).numpy()

    # errors
    rel_error = (costs_solution - costs_gt) / costs_gt
    error_thresh = ((costs(x_thresh) - costs_gt) / costs_gt)
    error_perm = ((costs_solution_perm - costs_gt) / costs_gt).item()
    perm_dist = max(np.sum(x_thresh@np.ones([n, 1]) == 0), np.sum(np.ones([1,n])@x_thresh == 0))
    
    # save result
    d = {}
    d['size'] = n
    d['result'] = x_solution
    d['perm'] = is_perm
    d['cost'] = costs_gt
    d['time'] = elapsed
    d['cost_solution'] = costs_solution
    d['rel_error'] = rel_error
    d['error_perm'] = error_perm
    d['error_thresh'] = error_thresh
    d['perm_dist'] = perm_dist
    np.savez(os.path.join(output_folder, str(i) + '.npz'), **d)
    print("Finished "+ str(i)  + ". Rel error: " + str(rel_error) + " Is Permutation: " + str(is_perm), perm_dist, error_perm, error_thresh)