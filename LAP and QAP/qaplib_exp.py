import os
import time
import random
import torch 
import matplotlib.pyplot as plt
import numpy as np
import itertools
from qaplib_reader import read_qap, read_qap_solution
from softmax_optimization import opt_softmax

maxIter = 30000
gamma = 50.0
learning_rate = 1e-3
# Choose Path to save and path to Faust Dataset
output_folder = './qaplib_results'
qaplib_folder = './QAPLIB'
use_GPU = True

if not os.path.exists(output_folder):
    os.mkdir(output_folder)
    
instances = []
for file in os.listdir(qaplib_folder):
    if file.endswith(".sln"):
        filename = (file.split('.'))[0]
        instances.append(filename)
        
def generateA (X,Y):
    A = np.zeros([n,n,n,n])
    for i in range(n):
        for j in range(n):
            for p in range(n):
                for q in range(n):
                    A[i,j,p,q] = X[i,p]*Y[j,q]
                    
    A = np.reshape(A, [n**2, n**2])
    
    # normalize A
    A_orig = A
    if abs(A.max()) > 1:
        A = A/A.max()
    
    # convert to torch
    A = torch.from_numpy(A.astype(np.float32))
    A_orig = torch.from_numpy(A_orig.astype(np.float32))
    
    b = torch.zeros(n**2)
    
    return A, A_orig, b



random.shuffle(instances)

for file in instances:
    # generate A
    filename = file + '.dat'
    X,Y = read_qap(os.path.join(qaplib_folder, filename))
    X = np.array(X)
    Y = np.array(Y)
    
    # set dimensions
    n = X.shape[0]
    m = np.ceil(n/3)
    
    A, A_orig, b = generateA(X,Y)
    
    # init costs
    lam = torch.linalg.matrix_norm(A, ord=2)
    I = torch.eye(n**2,n**2)

    costs = lambda x,alpha: torch.dot(x.view(-1), (A-alpha*I)@x.view(-1) + (b+alpha)) 
    costs_orig = lambda x,alpha: torch.dot(x.view(-1), (A_orig-alpha*I)@x.view(-1) + (b+alpha)) 
    thresh = lambda x: (x > 0.5)
    
    # optimization
    t = time.time()
    x_result = opt_softmax(A, b, gamma, maxIter, learning_rate, n, use_GPU)
    elapsed = time.time() - t
    
    # result
    x_perm = thresh(x_result)
    cost_orig = costs_orig(x_perm.float(),0)
    
    # save result
    d = {}
    d['result'] = x_result
    d['perm'] = x_perm
    d['cost'] = cost_orig
    d['time'] = elapsed
    np.savez(os.path.join(output_folder, filename + '.npz'), **d)