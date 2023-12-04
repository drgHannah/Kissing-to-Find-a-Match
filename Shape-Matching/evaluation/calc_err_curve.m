%***************************************************************************************
%*    This code is taken from https://github.com/riccardomarin/Diff-FMAPs-PyTorch
%*    of Riccardo Marin, Code version: 3f9e65c0aed822a1873f3dfd34485e5bb9342286
%***************************************************************************************

function curve = calc_err_curve(errors, thresholds)
    npoints = size(errors,1);
    curve = zeros(1,length(thresholds));
    for i=1:length(thresholds)
        curve(i) = 100*sum(errors <= thresholds(i))./ npoints;
    end  
end