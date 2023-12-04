import torch
import numpy as np
import tqdm

norm = lambda x:  x/torch.sqrt(torch.sum(x**2,1))[:,None]
def add_random_idc(c_list,n,multi):
    np_idc = np.random.randint(0, high=n, size=(int(multi*len(c_list[0])),2), dtype=int)
    idc_z = (list(np_idc[:,0]),list(np_idc[:,1]))
    dub_list=list(zip(idc_z[0],idc_z[1])) 
    ndub = list(set(dub_list)-set(list(zip(c_list[0],c_list[1]))))
    new =  list(zip(*ndub)) 
    if new == []:
        return []
    return (list(new[0]),list(new[1]))

def sparse_training_m(V, W, idc_global, A_data, beta=20,multi=200):
    n=V.shape[0]
    idc_new = add_random_idc(idc_global,n,multi=multi)
    if idc_new == [] or multi ==0:
        insm1 = beta  * (torch.sum(norm(V)[idc_global[0]] * norm(W)[idc_global[1]], dim=1))
        P = torch.sparse_coo_tensor(torch.tensor(idc_global).cuda(), insm1,size=(n,n))
        soft_P = torch.sparse.softmax(P,dim=0)
        lossA = torch.sum(A_data*soft_P.coalesce().values())
        return lossA, reg(soft_P, dim = 0), reg(soft_P, dim = 1)
    else:
        insm1 = beta  * (torch.sum(norm(V)[idc_global[0]] * norm(W)[idc_global[1]], dim=1))
        insm2 = beta  * (torch.sum(norm(V)[idc_new[0]] * norm(W)[idc_new[1]], dim=1))
        insm=torch.cat((insm1,insm2),dim=0)
        idc_combi=(idc_global[0]+idc_new[0],idc_global[1]+idc_new[1])
        P = torch.sparse_coo_tensor(torch.tensor(idc_combi).cuda(), insm,size=(n,n))

        Az = torch.cat((A_data,torch.zeros_like(torch.tensor(idc_new[0])).cuda()),dim=0)
        Asparse = torch.sparse_coo_tensor(torch.tensor(idc_combi).cuda(), Az,size=(n,n)).detach()
        soft_P = torch.sparse.softmax(P,dim=0)
        AB=(Asparse.coalesce().values()*soft_P.coalesce().values())

    return torch.sum(AB), reg(soft_P, dim = 0), reg(soft_P, dim = 1)

def reg(P, dim = 0):
    return torch.sum((torch.sparse.sum(P,dim=dim).coalesce().values()-1)**2)

def lap_softmax(A, gamma, maxIter, learning_rate, n, m, use_GPU = False, idc_global=None,multi=1):
    beta=20*2
    norm = lambda x:  x/torch.sqrt(torch.sum(x**2,1))[:,None]
    ATB = lambda A,B: norm(A) @ norm(B).T
    x = lambda A,B,beta: torch.nn.functional.softmax(beta * (ATB(A,B)), dim=0)
    V = torch.nn.Parameter(torch.rand((n,m)).cuda(),requires_grad=True)
    W = torch.nn.Parameter(torch.rand((n,m)).cuda(),requires_grad=True)
    optimizer = torch.optim.Adam({V,W}, lr=learning_rate)
    pbar = tqdm.tqdm(range(maxIter))
    losses = []
    for i in pbar:
        optimizer.zero_grad()
        Ap,r1,r2 = sparse_training_m(V, W, idc_global, A, beta=beta*(i/maxIter),multi=multi)
        loss = Ap + gamma * (r1+r2)
        pbar.set_postfix({'loss': loss.item(), 'AP':Ap.item(), 'reg':(r1+r2).item()})
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    x_solution = x(V,W,beta).cpu().detach().numpy()
    return x_solution, V, W,losses

