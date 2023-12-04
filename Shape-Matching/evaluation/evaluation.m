%***************************************************************************************
%*    This code is taken and modified 
%*    from https://github.com/riccardomarin/Diff-FMAPs-PyTorch
%*    of Riccardo Marin, Code version: 3f9e65c0aed822a1873f3dfd34485e5bb9342286
%***************************************************************************************

args = argv();


% clear all;
addpath(genpath('../utils'));
addpath(genpath('../evaluation'));


id = (args{1}); 
disp(id)


load([strcat('../results/out_FAUST_noise_0.01_',id,'.mat')])%

% Set the evaluation couples (similar to rng(3))
src = [17,11,9,67,76,77,54,36,80,7, 81,79,24,83,100,78,52,51,8,28, 40,70,68,19,88,43,49,14,48,21, 3,44,27,65,63,39,56,98,61,35, 29,41,20,23,10,66,50,12,45,42, 74,92,55,4,53,86,25,1,99,82, 38,18,64,47,16,75,97,13,72,37, 95,62,32,15,85,60,59,71,22,87, 57,2,73,91,94,58,26,90,93,34, 46,5,6,33,96,89,30,84,69,31];
tar = [26,28,90,21,84,35,48,100,89,3,54,13,60,42,51,4,5,23,63,83, 17,99,40,72,24,87,82,39,88,32,16,41,34,74,62,61,73,12,1,8, 92,15,14,31,79,30,19,47,81,36,33,46,71,85,27,77,52,68,43,56, 10,64,94,11,45,50,70,20,93,2,22,9,7,86,38,96,37,75,59,65, 58,66,95,76,29,97,55,53,69,78,98,80,91,49,44,18,57,25,67,6];
assert(not(sum(src == tar)));

% Computing the matching for each couple
match = zeros(100,1000);
match_opt = zeros(100,1000);

for i = 1:size(src,2)
    phiM = [squeeze(basis(src(i),:,:))];
    phiN = [squeeze(basis(tar(i),:,:))];
    [match(i,:), match_opt(i,:)] = our_match(phiM, phiN,[1:1000]); 
    if(exist('desc'))
     descM = squeeze(desc(src(i),:,:));
     descN = squeeze(desc(tar(i),:,:));
     [match_desc(i,:)] = our_match_desc(phiM, phiN,descM, descN);
    else
        match_desc(i,:) = match_opt(i,:);
    end
end


%% Evaluation

mean_error = [];

% We load the geodesic distance matrix
% load('../utils/N_out.mat');
load('N_out.mat');

thr = [0:0.0001:0.5];

for i = 1:100
    idx_src = src(i); idx_tar = tar(i);
    
    M.VERT = squeeze(vertices_clean(src,:,:)); M.TRIV = double(faces); M.n = size(M.VERT,1); M.m = size(M.TRIV,1);
    N.VERT = squeeze(vertices_clean(tar,:,:)); N.TRIV = double(faces); N.n = size(N.VERT,1); N.m = size(N.TRIV,1);

    % D2 = load(strcat('path_to_geodesic distance matrices'));
    % D2.D = D2.D./max(max(D2.D));
    % dist_m = D2.D;

    dist_m = D; % approximated geodesic distance matrix from https://github.com/riccardomarin/Diff-FMAPs-PyTorch
    
    match_d = match_desc(i,:); match_o = match_opt(i,:);
    errors = compute_err(dist_m, [1:1000],[match_d', match_o']);
    if i == 1
        curves = compute_all_curves(errors,thr);
        mean_error = errors;
    else
        curves = curves + compute_all_curves(errors,thr);
        mean_error = mean_error + errors;
    end


end

mean_curves = curves/size(src,2);
mean_error = mean(mean_error/size(src,2));
disp(mean_error)
disp(strcat("e_prob: ",num2str(mean_error(1))))
disp(strcat("e_emb: ",num2str(mean_error(2))))


outfile = strcat('../results_py/error_',num2str(id),'.csv');
csvwrite(outfile,mean_error);
outfilemat = strcat('../results_py/errorcurve_',num2str(id),'.mat');
save(outfilemat,'mean_curves')
