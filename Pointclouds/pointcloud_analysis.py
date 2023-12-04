import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import os
import tqdm
import pandas as pd
import utils
import pointcloud_analysis as pa

norm = lambda pred: pred / torch.sqrt(torch.sum(pred**2,dim=1))[:,None]

SAVEPATH = "./save_final_nll2/"
READPATH = "./exp_final.csv"

class myNet(nn.Module):
    def __init__(self,tr_size = 3):
        super().__init__()
        self.fc2 = nn.Linear(tr_size, tr_size,bias=False)
    def forward(self, x):
        x = self.fc2(x) # Transformation
        return x

class myNetEmbedding(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return x
    
def create_data(N, seed = 0):
    torch.manual_seed(seed)
    x = torch.linspace(0, 1, N)
    y = torch.sin(x * np.pi) 
    z = torch.cos(x * np.pi) 
    points = norm(torch.cat((x[:,None],y[:,None],z[:,None]), dim=1))
    unknown_weight = utils.rotm()
    points2 = points.clone()@unknown_weight
    return points, points2, unknown_weight, 0


def create_online_data(N):
    kiss_dict = utils.get_kiss()
    m = int([k for k, v in kiss_dict.items() if v >= N][0])
    points=utils.uniform_hypersphere(m,N)
    # Create Rotation Matrix
    svd = torch.svd(torch.rand(m,m))
    print(torch.det((svd.U@svd.V.T)))
    svd.U[:,0] = svd.U[:,0] * torch.det((svd.U@svd.V.T))
    rotm=(svd.U@svd.V.T)
    points = torch.tensor(points)
    points2 = points@rotm
    return points, points2, rotm, 0


def create_labels(N, how_many_indices_are_known = 15, seed = 0):
    torch.manual_seed(seed)
    labels = -torch.ones(N, dtype=torch.long)
    trueIndices = torch.linspace(0,N-1,N).type(torch.long)
    knownIndices = torch.randperm(N)[:how_many_indices_are_known][:,None]
    labels[knownIndices] = trueIndices[knownIndices] 
    return labels, trueIndices

def save_plot(A,B,i):
    A = A.detach().cpu()
    B = B.detach().cpu()
    plt.figure()
    ax = plt.figure().add_subplot(projection='3d')
    ax.scatter(A[:,0],A[:,1],A[:,2], 'bx')
    ax.scatter(B[:,0],B[:,1],B[:,2], 'rx')
    ax.axis([-2,2,-2,2])
    plt.savefig(f"./{SAVEPATH}/{i}.png")
    plt.close()
    plt.close()


def get_function(in_str):
    if in_str == "relu":
        relu_function = lambda alpha,x: torch.log(1/(1-alpha) * F.relu(x - alpha)+1e-5)
        return relu_function
    elif in_str == "softplus":
        function = lambda alpha,x: (F.softplus(alpha*x, beta=10))
        return function
    elif in_str == "leaky":
        leaky_relu_function = lambda alpha,x:  1/(1-alpha) * F.leaky_relu(x - alpha)
        return leaky_relu_function
    elif in_str == "softmax":
        function = lambda alpha,x: (F.log_softmax(alpha * x,dim=-1))
        return function
    elif in_str == "softplus2":
        function = lambda alpha,x: 1/(1-alpha) * F.softplus(x - alpha, beta=10)
        return function
    else:
        print("No valid loss function.")

def stochastic_training(nonzero_indices, A, B):
    known_ind = len(nonzero_indices[1])
    one_entries = torch.sum(A[nonzero_indices[0]] * B[nonzero_indices[1]], dim=1)
    extra = 1
    shift = int(torch.randint(1, known_ind, (extra,)))
    new_entry = torch.roll(nonzero_indices[1], shift)
    zero_entries = torch.sum(A[nonzero_indices[0]] * B[new_entry], dim=1)
    stacked = torch.stack((one_entries,zero_entries),dim=1)
    return stacked


def train_model(points, points2, model, embedding, labels, learning_rate, stochastic, \
                epochs, alphain, alpha_increase = True, loss_str = "softmax", optimizerv="SGD", \
                seed = 0, device = 'cpu'):

    # set seed
    torch.manual_seed(seed)

    # to device
    points = points.to(device)
    points2 = points2.to(device)
    model = model.to(device)
    embedding = embedding.to(device)
    labels = labels.to(device)

    target_sparse = torch.zeros(sum(labels>-1)).long().to(device)

    # new loss function
    extrafunction = get_function(loss_str)
    new_loss_function = lambda alpha,x,target: F.nll_loss((extrafunction(alpha,x)), target, ignore_index=-1)

    # optimizer and  loss 
    if optimizerv == "SGD":
        try:
            optimizer = torch.optim.SGD(list(model.parameters())+ list(embedding.parameters()), lr=learning_rate)
        except:
            optimizer = torch.optim.SGD(list(model.parameters()), lr=learning_rate)
    else:
        try:
            optimizer = torch.optim.Adam(list(model.parameters())+ list(embedding.parameters()), lr=learning_rate)
        except:
            optimizer = torch.optim.Adam(list(model.parameters()), lr=learning_rate)
    
    loss_function = new_loss_function
    losses = []
    pbar = tqdm.tqdm(range(epochs))
    for epoch in pbar:

        # set alpha
        if alpha_increase:
            alpha = epoch/epochs * alphain
            if ("relu" in loss_str) or (alphain == 0.5):
                alpha = alpha * 2 - 0.5

        else:
            alpha = alphain

        # calculate loss
        A = embedding(points)
        B = embedding(model(points2))

        if stochastic == True:
            cond = labels>-1
            allind = torch.arange(len(labels))[cond]
            labelsin = labels[cond]
            perm =stochastic_training([allind, labelsin],norm(A), norm(B))
            loss = loss_function(alpha,perm,target_sparse)
        else:
            perm = (norm(A) @ norm(B).T)
            loss = loss_function(alpha, perm, labels)

        # optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        pbar.set_postfix({'loss': loss.item()})

    return model.cpu() ,embedding.cpu() , losses


def validate_model(model, embedding, points, points2, trueIndices, losses, alpha, setting = None, 
                    id = -1, loss_str = "softmax", permutedinput = None):
    model.eval()
    sm = torch.nn.Softmax(dim=1)
    A = (points)
    B = (model(points2))

    Aemb = embedding(points)
    Bemb = embedding(model(points2))

    # new loss function
    extrafunction = get_function(loss_str)
    new_loss_function = lambda x,target: F.nll_loss((extrafunction(alpha,x)), target)

    perm = torch.exp(extrafunction(alpha , (norm(Aemb) @ norm(Bemb).T)))
    binary_perm = utils.argmax_matrix(perm).float()

    # new loss
    newloss = new_loss_function((norm(Aemb) @ norm(Bemb).T), trueIndices)

    # average k for nearest neighbor
    D = torch.cdist(norm(A), norm(B))
    closest = torch.argsort(D, axis=1)
    poss = []
    for i in range(closest.shape[0]):
        pos = torch.where(closest[i] == trueIndices[i])[0].item()
        poss.append(pos)
    position = torch.tensor(poss).float().mean()

    # binary
    binary = torch.sum(binary_perm.int() - torch.nn.functional.one_hot(trueIndices)) # correctness

    # distance of corresponding point
    if permutedinput is not None:
        point_distance = torch.norm(norm(A)[permutedinput] - norm(B)) / A.shape[0]

    # Save Settings
    if setting is not None:
        df = pd.DataFrame(setting,index=[0])
    else:
        df = pd.DataFrame(index=[0])

    # Save Accuracies
    df["position"] = position.item()
    df["point_distance"] = point_distance.item()
    df["binary"] = binary.item()
    df["newloss"] = newloss.item()
    df["id"] = id



    return df, {"perm": perm.detach(), "A": (A), "B": (B), "loss": losses}


if __name__ == "__main__":

    dfs = []
    for N in [10000,1000,100,10]:
        number_known = N
        seed = 0
        loss = "softmax" # choose softmax or relu
        alpha = 1000
        epochs = 20000
        lr = 0.01
        increase_alpha = 1
        opt = "Adam"
        stoch = 1
        embedding = myNetEmbedding()
        
        for seed in torch.arange(0,3,1):

            pointso, points2o, unknown_weight, losses = pa.create_online_data(N)

            # permute data
            randpermutation = torch.randperm(N)#.sort(descending=False)[0]
            inverse_randpermutation = np.argsort(randpermutation)

            # apply permutation to points
            points2 = points2o
            points = pointso[randpermutation]

            # create and permute labels
            labels, trueIndices = pa.create_labels(N, number_known, seed)
            labels = labels[randpermutation]
            trueIndices = trueIndices[randpermutation]


            model = myNet(tr_size=points2.shape[1])

            trainedModel, embedding, losses = pa.train_model(points, points2, model, embedding, labels, lr, \
                                                            stoch, epochs,alpha, increase_alpha, loss, opt, seed = seed, device = 'cuda')

            df, PABloss = pa.validate_model(trainedModel, embedding, points, points2, trueIndices, losses, alpha, setting = None, 
                                        id=0, loss_str=loss, permutedinput=inverse_randpermutation)


            dfs.append(df)

    
    df = pd.concat(dfs)
    print(df)
    print(df.mean())
    print(df.std())


