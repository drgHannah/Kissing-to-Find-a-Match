%***************************************************************************************
%*    This code is taken from https://github.com/riccardomarin/Diff-FMAPs-PyTorch
%*    of Riccardo Marin, Code version: 3f9e65c0aed822a1873f3dfd34485e5bb9342286
%***************************************************************************************

function [match] = our_match_desc(phiM,phiN, descM, descN)
    F = pinv(phiM)*descM; %T
    G = pinv(phiN)*descN; %C
    C = (F'\G')'; %T\C
    
    match = knnsearch(phiN*C, phiM,1);

    
    %     match = knnsearch(phiN*C', phiM); 0,54
    %     match = knnsearch(phiN, phiM*C) 0,6
    %     match = knnsearch(phiN, phiM*C'); 0,5
    %     match = knnsearch(phiN*C, phiM); worst