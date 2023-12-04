"""***************************************************************************************
*    This code is taken and modified 
*    from https://github.com/riccardomarin/Diff-FMAPs-PyTorch
*    of Riccardo Marin, Code version: 3f9e65c0aed822a1873f3dfd34485e5bb9342286
***************************************************************************************/"""

from model import PointNetBasis
from model import PointNetDesc as PointNetDesc
from dataload import Surr12kModelNetDataLoader as DataLoader
import dataload
import torch
import numpy as np
import hdf5storage 
import glob
import scipy 
from get_id import get_train_type
import argparse
base = '../'

parser = argparse.ArgumentParser()
parser.add_argument('--current_time')
parser.add_argument('--experiment')
argsct = parser.parse_args()
current_time = argsct.current_time
experiment = argsct.experiment
print('(Test Faust) ID: ', current_time)
pretrained = False


device = torch.device("cpu")
DATA_PATH = base+'/data/FAUST_noise_0.01.mat'


basis_model = PointNetBasis(k=20, feature_transform=False)
desc_model = PointNetDesc(k=40, feature_transform=False)

print(base+f'/models/trained/{experiment}/{current_time}_basis_model_best_*.pth')
checkpoint = torch.load(glob.glob(base+f'/models/trained/{experiment}/{current_time}_basis_model_best_*.pth')[0])
basis_model.load_state_dict(checkpoint)

checkpoint = torch.load(glob.glob(base+f'/models/trained/{experiment}/{current_time}_desc_model_best_*.pth')[0])
desc_model.load_state_dict(checkpoint)

basis_model = basis_model.eval()
desc_model = desc_model.eval()

# Loading Data
dd = hdf5storage.loadmat(DATA_PATH)
v = dd['vertices'].astype(np.float32)

# Computing Basis and Descriptors
pred_basis = basis_model(torch.transpose(torch.from_numpy(dd['vertices'].astype(np.float32)),1,2))#vertices_clean vertices
pred_desc = desc_model(torch.transpose(torch.from_numpy(dd['vertices'].astype(np.float32)),1,2))

# Save Output
dd['basis'] = np.squeeze(np.asarray(pred_basis[0].detach().numpy()))
dd['desc'] = np.squeeze(np.asarray(pred_desc[0].detach().numpy()))

import os
os.makedirs(base+"/results/", exist_ok=True)
scipy.io.savemat(base+f'/results/out_FAUST_noise_0.01_{current_time}.mat', dd)
    