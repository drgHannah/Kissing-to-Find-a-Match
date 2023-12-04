from scipy.io import loadmat
import numpy as np

def open_faust_mat(filename):
    faust = loadmat(filename)
    
    hks = faust['hks']
    shot = faust['shot']
    
    desc = np.concatenate((hks, shot), axis=1)
    
    return desc