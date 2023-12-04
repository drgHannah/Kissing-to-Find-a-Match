import torch
import numpy as np


def lap_softmax(A, gamma, maxIter, learning_rate, n, m, use_GPU = False):
    I = torch.eye(n,n)
    beta=20

    norm = lambda x:  x/torch.sqrt(torch.sum(x**2,1))[:,None]
    ATB = lambda A,B: norm(A) @ norm(B).T
    x = lambda A,B: torch.nn.functional.softmax(beta * 2 * (ATB(A,B)), dim=0)

    V = torch.nn.Parameter(torch.rand((n,m)),requires_grad=True)
    W = torch.nn.Parameter(torch.rand((n,m)),requires_grad=True)
    
    device = torch.device("cuda" if use_GPU else "cpu")
    if use_GPU:
        V.to(device)
        W.to(device)

    optimizer = torch.optim.Adam({V,W}, lr=learning_rate)
    
    for i in range(maxIter):
        
        constraintPenalty = torch.sum((torch.sum(x(V,W),dim=0)-1)**2) + torch.sum((torch.sum(x(V,W),dim=1)-1)**2)
        loss = torch.trace(A @ x(V,W)) + gamma*constraintPenalty
    
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    x_solution = x(V,W).detach().numpy()
    return x_solution

def opt_softmax(A, b, gamma, maxIter, learning_rate, n, use_GPU = False):
    I = torch.eye(n**2,n**2)

    beta = 20

    norm = lambda x:  x/torch.sqrt(torch.sum(x**2,1))[:,None]
    ATB = lambda A,B: norm(A) @ norm(B).T
    x = lambda A,B: torch.nn.functional.softmax(beta * 2 * (ATB(A,B)))

    m = n-1
    
    device = torch.device("cuda" if use_GPU else "cpu")

    V = torch.nn.Parameter(torch.rand((n,m)),requires_grad=True)
    W = torch.nn.Parameter(torch.rand((n,m)),requires_grad=True)
    
    if use_GPU:
        V.to(device)
        W.to(device)

    optimizer = torch.optim.Adam({V,W}, lr=learning_rate)
    losses = []

    lam = torch.linalg.matrix_norm(A, ord=2)
    a = torch.linspace(-lam,lam,maxIter)
    for i in range(maxIter):
        alpha = a[i]

        constraintPenalty = torch.sum((torch.sum(x(V,W),dim=0)-1)**2) + torch.sum((torch.sum(x(V,W),dim=1)-1)**2)
        loss = torch.dot(x(V,W).view(-1), (A-alpha*I)@x(V,W).view(-1) + (b+alpha))   +  gamma*constraintPenalty

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    x_relu = x(V,W).detach().cpu()
    
    return x_relu