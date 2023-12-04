import torch 
import matplotlib.pyplot as plt
import numpy as np
import itertools
import os
import scipy.io
from faustDS_reader import open_faust_mat


# set dimensions
n = 100
m = 30

maxIter = 50000
gamma = 30.0
learning_rate = 1e-3

# Choose Path to save and path to Faust Dataset
output_folder = 'faust_results'
faust_folder = './data/'
use_GPU = True

subsample = np.arange(n)

if not os.path.exists(output_folder):
    os.mkdir(output_folder)
    
    
instances = []
for file in os.listdir(faust_folder):
    if file.endswith(".mat"):
        filename = (file.split('.'))[0]
        instances.append(filename)
        
n_instances = len(instances)
        
def generateA (X,Y,ks):
    A = X[ks,:] @ Y[ks,:].T
    
    A = -A
    
    # normalize A
    if abs(A.max()) > 1:
        A = A/A.max()
    
    # convert to torch
    A = torch.from_numpy(A.astype(np.float32))
    
    return A

from scipy.optimize import linear_sum_assignment
from softmax_optimization import lap_softmax
import time

perms = []
rel_errors = []

thresh = lambda x: (x > 0.5)
def argmaxing(x):
    x_argmax = np.zeros(x.shape)
    argmax_inds = x.argmax(axis=1)
    x_argmax[np.arange(x.shape[0]),argmax_inds] = 1
    return x_argmax

for i in range(n_instances):
    print("Started " + instances[i])
    # set up instance
    random_match = np.random.randint(0,n_instances)
    descX = open_faust_mat(os.path.join(faust_folder, instances[i] + '.mat'))
    descY = open_faust_mat(os.path.join(faust_folder, instances[random_match] + '.mat'))
    A = generateA(descX, descY,subsample)

    if abs(A.max()) > 1:
        A = A/A.max()

    costs = lambda x: torch.sum(A * x)

    # LAP
    solution = linear_sum_assignment(A)

    x_gt = np.zeros([n, n])
    for j in solution[0]:
        x_gt[j, solution[1][j]] = 1
        
    # optimization
    t = time.time()
    x_solution = lap_softmax(A, gamma, maxIter, learning_rate, n, m, use_GPU)
    elapsed = time.time() - t
    
    # evaluation
    costs_gt = costs(x_gt).numpy()
    
    #x_thresh = thresh(x_solution)
    x_thresh = argmaxing(x_solution)

    is_perm = False
    if (x_thresh@np.ones([n, 1]) == 1).all() and (np.ones([1,n])@x_thresh == 1).all():
        is_perm = True
        
    print(x_thresh@np.ones([n, 1]))
    print(np.ones([1,n])@x_thresh)
        
    costs_solution = costs(x_solution).numpy()

    rel_error = (costs_solution - costs_gt) / costs_gt
    
    # save result
    d = {}
    d['result'] = x_solution
    d['perm'] = is_perm
    d['cost'] = costs_gt
    d['time'] = elapsed
    d['cost_solution'] = costs_solution
    d['rel_error'] = rel_error
    np.savez(os.path.join(output_folder, instances[i] + '.npz'), **d)
    print("Finished " + instances[i] + ". Rel error: " + str(rel_error) + " Is Permutation: " + str(is_perm))