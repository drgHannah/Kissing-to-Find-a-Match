%***************************************************************************************
%*    This code is taken from https://github.com/riccardomarin/Diff-FMAPs-PyTorch
%*    of Riccardo Marin, Code version: 3f9e65c0aed822a1873f3dfd34485e5bb9342286
%***************************************************************************************

function dist_m = normDistMatr(N)
    dist_m = calc_dist_matrix(N,[1:N.n]);
    diam = max(max(dist_m));
    dist_m = dist_m./diam;