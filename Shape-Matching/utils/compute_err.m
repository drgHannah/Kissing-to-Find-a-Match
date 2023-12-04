%***************************************************************************************
%*    This code is taken from https://github.com/riccardomarin/Diff-FMAPs-PyTorch
%*    of Riccardo Marin, Code version: 3f9e65c0aed822a1873f3dfd34485e5bb9342286
%***************************************************************************************

function [err] = compute_err(dist_m,gt_match, matches)
    for i = 1:size(matches,2)
        for j = 1:size(matches,1)
            err(j,i) = dist_m(gt_match(j),matches(j,i));
        end
    end