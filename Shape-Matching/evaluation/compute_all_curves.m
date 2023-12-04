%***************************************************************************************
%*    This code is taken from https://github.com/riccardomarin/Diff-FMAPs-PyTorch
%*    of Riccardo Marin, Code version: 3f9e65c0aed822a1873f3dfd34485e5bb9342286
%***************************************************************************************

function c = compute_all_curves(err,thr)
    for i=1:size(err,2)
        c(i,:) = calc_err_curve(err(:,i),thr);
    end