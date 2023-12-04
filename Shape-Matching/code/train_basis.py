"""***************************************************************************************
*    This code is taken and modified 
*    from https://github.com/riccardomarin/Diff-FMAPs-PyTorch
*    of Riccardo Marin, Code version: 3f9e65c0aed822a1873f3dfd34485e5bb9342286
***************************************************************************************/"""

from __future__ import print_function
import argparse
import os
import random
import torch
import torch.optim as optim
from model import PointNetBasis
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
from dataload import Surr12kModelNetDataLoader as DataLoader
from dataload import Tosca_DataLoader as DataLoader_tosca
from dataload import MyBatchSampler
from tqdm import tqdm
from get_id import get_train_type
import matplotlib.pyplot as plt
# from plot import plot
import time
from train_desc import train_desc

import setuptools
print(setuptools.__version__)
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime


print('Train Basis', datetime.now())

args = get_train_type()
train_type = args.train_type
# b = args.b
name_str = args.name
in_epoch = args.epoch
dataset_name = args.dataset
npoint = args.npoint



now = datetime.now()
current_time = name_str  
print("Current Time =", current_time)

lr = args.lr
no_vertices_scaling = 1 / args.npoint * 1000

# Create writer
summarypath = f"{args.name}/"+current_time
writer = SummaryWriter(comment=summarypath)

checkpoint_continue = args.pretrain

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("device: ", device)
manualSeed = 1  # fix seed
random.seed(manualSeed)
torch.manual_seed(manualSeed)

b_size = 8
hp = {'train_type': train_type, 'lr': lr, 'checkpoint_continue': checkpoint_continue, 'id': current_time}
writer.add_text('lstm', str(hp), 0)




# get the start time
st = time.time()




# Out Dir
outf = f'./models/trained/{name_str}'
try:
    os.makedirs(outf)
except OSError:
    pass

DATA_PATH = 'data/'
if dataset_name == 'surr12k':
    TRAIN_DATASET = DataLoader(root=DATA_PATH, npoint=npoint, split='train', normal_channel=False, augm = True)
    TEST_DATASET = DataLoader(root=DATA_PATH, npoint=npoint, split='test', normal_channel=False, augm = True)
elif dataset_name == 'tosca':
    TRAIN_DATASET = DataLoader_tosca(root=DATA_PATH, npoint=npoint, uniform=False, augm=True, split = "train")
    TEST_DATASET = DataLoader_tosca(root=DATA_PATH, npoint=npoint, uniform=False, augm=True, split = "test")

if dataset_name == 'surr12k':
    dataset = torch.utils.data.DataLoader(TRAIN_DATASET, batch_size=b_size, shuffle=True, num_workers=8)
    dataset_test = torch.utils.data.DataLoader(TEST_DATASET, batch_size=b_size, shuffle=True, num_workers=8)
elif dataset_name == 'tosca':
    my_sampler_train = MyBatchSampler(TRAIN_DATASET, batch_size=b_size)
    dataset = torch.utils.data.DataLoader(TRAIN_DATASET, batch_sampler=my_sampler_train)
    my_sampler_test = MyBatchSampler(TEST_DATASET, batch_size=2)
    dataset_test = torch.utils.data.DataLoader(TEST_DATASET, batch_sampler=my_sampler_test)


basisNet = PointNetBasis(k=20, feature_transform=False)

if checkpoint_continue:
    try:
        print("Load pretrained network...")
        checkpoint = torch.load('./models/pretrained/basis_model_best.pth')
        basisNet.load_state_dict(checkpoint)
        basisNet.cuda()
    except:
        print(f"Could not load checkpoint './models/pretrained/basis_model_best.pth' - continue without loading....")


# Optimizer
optimizer = optim.Adam(basisNet.parameters(), lr=lr, betas=(0.9, 0.999))


basisNet.cuda()

best_eval_loss = np.inf

train_losses = []
eval_losses = []
faust_losses = []


# Training Loop

def col_norm(A):
    return (A/torch.sqrt(torch.sum(A**2, 1)[:,None]))


def row_norm(A):
    return (A/torch.sqrt(torch.sum(A**2, 2)[:,:,None]))


def dist(X, Y):
    sx = torch.sum(X**2, dim=2, keepdims=True) #rows: l2-norm^2
    sy = torch.sum(Y**2, dim=2, keepdims=True) #rows: l2-norm^2
    return torch.sqrt(-2 * torch.matmul(X,Y.permute(0,2,1)) + sx + sy.permute(0,2,1))


def dist_nosqrt(X, Y):
    sx = torch.sum(X**2,dim=2,keepdims=True) #rows: l2-norm^2
    sy = torch.sum(Y**2,dim=2,keepdims=True) #rows: l2-norm^2
    return (-2 * torch.matmul(X,Y.permute(0,2,1)) + sx + sy.permute(0,2,1))


def pinverse(basis_A):
    return torch.inverse(basis_A.transpose(1,2)@basis_A) @ basis_A.transpose(1,2)

if checkpoint_continue:
    epochs = in_epoch
else:
    epochs = in_epoch #1600 #400


torch.cuda.empty_cache()
for epoch in tqdm(range(epochs)):

    train_loss = 0
    train_loss_iterator = 0

    for data in tqdm(dataset, 0, disable=True):

        points = data[0]
        points = points.transpose(2, 1)
        points = points.cuda().to(device)

        optimizer.zero_grad()
        
        basisNet = basisNet.train()
        # Obtaining predicted basis
        pred, _, _ = basisNet(points)

        # Generating pairs
        basis_A = pred[1:, :, :]
        basis_B = pred[:-1, :, :]
        pc_A = points[1:, :, :]
        pc_B = points[:-1, :, :]



        # Marin etal
        if train_type == 0:
            # Computing optimal transformation
            pseudo_inv_A = pinverse(basis_A)
            C_opt = torch.matmul(pseudo_inv_A, basis_B)
            opt_A = torch.matmul(basis_A, C_opt)

            # SoftMap
            dist_matrix = torch.cdist(opt_A, basis_B)
            s_max = torch.nn.Softmax(dim=1)
            s_max_matrix = s_max(-dist_matrix)

            # Basis Loss
            eucl_loss = torch.sum(torch.square(torch.matmul(s_max_matrix, torch.transpose(pc_B,1,2)) - torch.transpose(pc_B,1,2))) * no_vertices_scaling

        # Standard Training
        if train_type == 1:

            pseudo_inv_A = pinverse(basis_A)
            C_opt = torch.matmul(pseudo_inv_A, basis_B)
            opt_A = torch.matmul(basis_A, C_opt)

            # SoftMap
            dist_matrix = torch.cdist(row_norm(opt_A), row_norm(basis_B))
            s_max = torch.nn.Softmax(dim=2)
            s_max_matrix = s_max(-20*dist_matrix)

            # Basis Loss
            eucl_loss = torch.sum(torch.square(torch.matmul(s_max_matrix, torch.transpose(pc_B, 1, 2)) - torch.transpose(pc_B, 1, 2))) * no_vertices_scaling

        # Stochastic Training Variant
        if train_type == 2:
            number_extra_entries = args.number_extra_entries
            pc_B_here = pc_B.permute(0, 2, 1)

            pseudo_inv_A = pinverse(basis_A)
            C_opt = torch.matmul(pseudo_inv_A, basis_B)
            opt_A = row_norm(torch.matmul(basis_A, C_opt))
            basis_B = row_norm(basis_B)

            # diagonal matrix and shifts
            diagonal = (opt_A * basis_B).sum(2, keepdim=True)
            shifts = torch.randperm(points.shape[-1] - 1)[:number_extra_entries] + 1

            # get permutation matrix entries
            for sk in shifts:
                k = (basis_B[:, sk:, :] * opt_A[:, :-sk, :]).sum(2, keepdim=True)
                j = (basis_B[:, :sk, :] * opt_A[:, -sk:, :]).sum(2, keepdim=True)
                n = (torch.cat((k, j), dim=1))
                diagonal = torch.cat((diagonal, n), dim=2)

            # calculate permutation matrix
            dist_matrix = (-2 * (diagonal) + 2)
            s_max_matrix = torch.nn.functional.softmax(-dist_matrix * 20, dim=2)

            # permute point values
            pc_Bs = pc_B_here.permute(2, 0, 1)[None]  # None,xyz,b,i
            for sk in shifts:
                k = (pc_B_here.permute(2, 0, 1)[:, :, sk:])
                j = (pc_B_here.permute(2, 0, 1)[:, :, :sk])
                n = (torch.cat((k, j), dim=2))[None]
                pc_Bs = torch.cat((pc_Bs, n), dim=0)

            # permute mult(X_new,p)
            mult_xyz = (s_max_matrix * pc_Bs.permute(1, 2, 3, 0)).sum(-1).permute(1, 2, 0)

            # loss
            eucl_loss =  torch.sum(torch.square(mult_xyz - pc_B_here)) * no_vertices_scaling


        # Back Prop
        train_loss += eucl_loss.item()
        train_loss_iterator += 1
        eucl_loss.backward()
        optimizer.step()

    print("validation")
    eval_loss = 0
    eval_loss_iterator=1
    if True:
        with torch.no_grad():
            eval_loss = 0
            eval_loss_iterator = 0
            for data in tqdm(dataset_test, 0, disable=True):#_test
                points = data[0]

                points = points.transpose(2, 1)
                points = points.cuda()
                basisNet = basisNet.eval()
                pred, _, _ = basisNet(points)

                basis_A = pred[1:, :, :]
                basis_B = pred[:-1, :, :]
                pc_A = points[1:, :, :]
                pc_B = points[:-1, :, :]


                # Marin etal
                if train_type == 0:

                    # Computing optimal transformation
                    pseudo_inv_A = pinverse(basis_A)
                    C_opt = torch.matmul(pseudo_inv_A, basis_B)
                    opt_A = torch.matmul(basis_A, C_opt)

                    # SoftMap
                    dist_matrix = torch.cdist(opt_A, basis_B)
                    s_max = torch.nn.Softmax(dim=1)
                    s_max_matrix = s_max(-dist_matrix)

                    # Basis Loss
                    eucl_loss = torch.sum(torch.square(torch.matmul(s_max_matrix, torch.transpose(pc_B,1,2)) - torch.transpose(pc_B,1,2)))

                # Standard Training
                if train_type == 1:

                    # Computing optimal transformation
                    pseudo_inv_A = pinverse(basis_A)
                    C_opt = torch.matmul(pseudo_inv_A, basis_B)
                    opt_A = torch.matmul(basis_A, C_opt)

                    # SoftMap
                    dist_matrix = torch.cdist(row_norm(opt_A), row_norm(basis_B))
                    s_max = torch.nn.Softmax(dim=2)
                    s_max_matrix = s_max(-20*dist_matrix)

                    # Basis Loss
                    eucl_loss = torch.sum(torch.square(torch.matmul(s_max_matrix, torch.transpose(pc_B,1,2)) - torch.transpose(pc_B,1,2))) * no_vertices_scaling


                # Stochastic Training Variant
                if train_type == 2:
                    number_extra_entries = max(args.number_extra_entries,1)
                    pc_B_here = pc_B.permute(0, 2, 1)

                    # Computing optimal transformation
                    pseudo_inv_A = pinverse(basis_A)
                    C_opt = torch.matmul(pseudo_inv_A, basis_B)
                    opt_A = row_norm(torch.matmul(basis_A, C_opt))
                    basis_B = row_norm(basis_B)

                    # diagonal matrix and shifts
                    diagonal = (opt_A * basis_B).sum(2, keepdim=True)
                    shifts = torch.randperm(points.shape[-1] - 1)[:number_extra_entries] + 1

                    # get permutation matrix entries
                    for sk in shifts:
                        k = (basis_B[:, sk:, :] * opt_A[:, :-sk, :]).sum(2, keepdim=True)
                        j = (basis_B[:, :sk, :] * opt_A[:, -sk:, :]).sum(2, keepdim=True)
                        n = (torch.cat((k, j), dim=1))
                        diagonal = torch.cat((diagonal, n), dim=2)

                    # calculate permutation matrix
                    dist_matrix = (-2 * (diagonal) + 2)
                    s_max_matrix = torch.nn.functional.softmax(-dist_matrix * 20, dim=2)

                    # permute point values
                    pc_Bs = pc_B_here.permute(2, 0, 1)[None]  # None,xyz,b,i
                    for sk in shifts:
                        k = (pc_B_here.permute(2, 0, 1)[:, :, sk:])
                        j = (pc_B_here.permute(2, 0, 1)[:, :, :sk])
                        n = (torch.cat((k, j), dim=2))[None]
                        pc_Bs = torch.cat((pc_Bs, n), dim=0)

                    # permute mult(X_new,p)
                    mult_xyz = (s_max_matrix * pc_Bs.permute(1, 2, 3, 0)).sum(-1).permute(1, 2, 0)

                    # loss
                    eucl_loss = torch.sum(torch.square(mult_xyz - pc_B_here)) * no_vertices_scaling


                eval_loss +=   eucl_loss.item()
                eval_loss_iterator += 1
            print('EPOCH ' + str(epoch) + ' - eva_loss: ' + str(eval_loss/eval_loss_iterator))

    print('EPOCH ' + str(epoch) + ' - train_loss: ' + str(train_loss/train_loss_iterator))


    # Saving if best model so far
    if eval_loss < best_eval_loss:
        print('save model')
        best_eval_loss = eval_loss
        torch.save(basisNet.state_dict(), f'{outf}/{current_time}_basis_model_best_{train_type}_0.pth')

    train_losses.append(train_loss/train_loss_iterator)
    eval_losses.append(eval_loss/eval_loss_iterator)

    writer.add_scalar('Loss/train', train_losses[-1], epoch)
    writer.add_scalar('Loss/test', eval_losses[-1], epoch)


    # Logging losses
    np.save(outf+f'/{current_time}_train_losses_basis_{train_type}_0.npy',train_losses)
    np.save(outf+f'/{current_time}_eval_losses_basis_{train_type}_0.npy',eval_losses)

    writer.close()

# get the end time
et = time.time()

# get the execution time
elapsed_time = et - st
print('Execution time:', elapsed_time, 'seconds')
print("Train Desc Now")
train_desc(args,current_time)