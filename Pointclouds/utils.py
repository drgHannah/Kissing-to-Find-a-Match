
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
from itertools import product, combinations
import torch

def get_rgb_cycle(leng):
    phi = np.linspace(0, 2*np.pi, leng)
    rgb_cycle = (np.stack((np.cos(phi), np.cos(phi+2*np.pi/3), np.cos(phi-2*np.pi/3))).T + 1)*0.5                        
    return rgb_cycle


def dimension_reduction(A,dimension = 2):
    return A@torch.pca_lowrank(A, q=dimension, center=True, niter=10)[2]


def argmax_matrix(input):
    return torch.nn.functional.one_hot(torch.argmax(input,dim=-1), num_classes=input.shape[-1])

def plot_wireframe(points,points2,name="tmp.png"):

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

    # draw cube
    r = [-1, 1]
    for s, e in combinations(np.array(list(product(r, r, r))), 2):
        if np.sum(np.abs(s-e)) == r[1]-r[0]:
            ax.plot3D(*zip(s, e), color="b", linewidth=0.2)

    # draw sphere
    u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
    x = np.cos(u)*np.sin(v)
    y = np.sin(u)*np.sin(v)
    z = np.cos(v)
    ax.plot_wireframe(x, y, z, color="black", linewidth=0.2)

    # draw a point
    rgb = get_rgb_cycle(points.shape[0])
    ax.scatter(points[:,0], points[:,1], points[:,2], color=rgb, s=20) #  "r"
    ax.scatter(points2[:,0], points2[:,1], points2[:,2], color=rgb, s=20, marker = 'v')
    
    ax.grid(False)
    plt.show()
    # plt.savefig(name, bbox_inches = 'tight',
    # pad_inches = 0)

def get_angles(x2,x1=[0,0,1]):
    x1 = torch.tensor([x1])
    distance = torch.sqrt((x2[:,0]-x1[:,0])**2+(x2[:,1]-x1[:,1])**2+(x2[:,2]-x1[:,2])**2)
    plunge = torch.rad2deg(torch.asin(x2[:,2]-x1[:,2]/distance))
    azimuth = torch.rad2deg(torch.atan2(x2[:,0]-x1[:,0], x2[:,1]-x1[:,1]))
    return plunge, azimuth

def rotm():
    roll = yaw = pitch = torch.randn(1)
    cos = torch.cos
    sin = torch.sin
    RX = torch.tensor([
                    [1, 0, 0],
                    [0, cos(roll), -sin(roll)],
                    [0, sin(roll), cos(roll)]
                ])
    RY = torch.tensor([
                    [cos(pitch), 0, sin(pitch)],
                    [0, 1, 0],
                    [-sin(pitch), 0, cos(pitch)]
                ])
    RZ = torch.tensor([
                    [cos(yaw), -sin(yaw), 0],
                    [sin(yaw), cos(yaw), 0],
                    [0, 0, 1]
                ])
    R = torch.mm(RZ, RY)
    R = torch.mm(R, RX)
    return R

def get_kiss():
    kissing = {
        "1":2,
        "2" :	6,
        "3" :	12,
        "4" :	24,
        "5" :	40,
        "6" :	72,
        "7" :	126,
        "8" :	240,
        "9" :	306,
        "10" :	500,
        "11" :	582,
        "12" :	840,
        "13" :	1154,
        "14" :	1606,
        "15" :	2564,
        "16" :	4320,
        "17" :	5346,
        "18" :	7398,
        "19" :	10668,
        "20" :	17400,
        "21" :	27720,
        "22" :	49896,
        "23" :	93150,
        "24" :	196560,
    }
    return kissing


#####################################################################################################################################
### The following code is taken from ################################################################################################
### https://stackoverflow.com/questions/57123194/how-to-distribute-points-evenly-on-the-surface-of-hyperspheres-in-higher-dimensi ###
#####################################################################################################################################

from itertools import count, islice
from math import cos, gamma, pi, sin, sqrt
from typing import Callable, Iterator, List

def int_sin_m(x: float, m: int) -> float:
    """Computes the integral of sin^m(t) dt from 0 to x recursively"""
    if m == 0:
        return x
    elif m == 1:
        return 1 - cos(x)
    else:
        return (m - 1) / m * int_sin_m(x, m - 2) - cos(x) * sin(x) ** (
            m - 1
        ) / m

def primes() -> Iterator[int]:
    """Returns an infinite generator of prime numbers"""
    yield from (2, 3, 5, 7)
    composites = {}
    ps = primes()
    next(ps)
    p = next(ps)
    assert p == 3
    psq = p * p
    for i in count(9, 2):
        if i in composites:  # composite
            step = composites.pop(i)
        elif i < psq:  # prime
            yield i
            continue
        else:  # composite, = p*p
            assert i == psq
            step = 2 * p
            p = next(ps)
            psq = p * p
        i += step
        while i in composites:
            i += step
        composites[i] = step

def inverse_increasing(
    func: Callable[[float], float],
    target: float,
    lower: float,
    upper: float,
    atol: float = 1e-10,
) -> float:
    """Returns func inverse of target between lower and upper

    inverse is accurate to an absolute tolerance of atol, and
    must be monotonically increasing over the interval lower
    to upper    
    """
    mid = (lower + upper) / 2
    approx = func(mid)
    while abs(approx - target) > atol:
        if approx > target:
            upper = mid
        else:
            lower = mid
        mid = (upper + lower) / 2
        approx = func(mid)
    return mid

def uniform_hypersphere(d: int, n: int) -> List[List[float]]:
    """Generate n points over the d dimensional hypersphere"""
    assert d > 1
    assert n > 0
    points = [[1 for _ in range(d)] for _ in range(n)]
    for i in range(n):
        t = 2 * pi * i / n
        points[i][0] *= sin(t)
        points[i][1] *= cos(t)
    for dim, prime in zip(range(2, d), primes()):
        offset = sqrt(prime)
        mult = gamma(dim / 2 + 0.5) / gamma(dim / 2) / sqrt(pi)

        def dim_func(y):
            return mult * int_sin_m(y, dim - 1)

        for i in range(n):
            deg = inverse_increasing(dim_func, i * offset % 1, 0, pi)
            for j in range(dim):
                points[i][j] *= sin(deg)
            points[i][dim] *= cos(deg)
    return points