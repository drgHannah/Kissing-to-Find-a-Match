%***************************************************************************************
%*    This code is taken and mod. from https://github.com/riccardomarin/Diff-FMAPs-PyTorch
%*    of Riccardo Marin, Code version: 3f9e65c0aed822a1873f3dfd34485e5bb9342286
%***************************************************************************************

function [match, match_opt] = our_match(phiM,phiN, gt_match,gt_matchM)
    match = 0;%knnsearch(phiN,phiM,1);
    if exist('gt_matchM')
        C = phiM(gt_matchM,:)\phiN(gt_match,:);
    else
        C = phiM\phiN(gt_match,:);
    end
    match_opt = knnsearch(phiN,phiM*C,1);

