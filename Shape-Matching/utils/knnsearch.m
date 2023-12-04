

function ind_closest = knnsearch(x,newpoint,k)
 
Q = size(newpoint, 1);
M = size(x, 2);


nA = sum(newpoint.^2, 2); %// Sum of squares for each row of A
nB = sum(x.^2, 2); %// Sum of squares for each row of B
D = bsxfun(@plus, nA, nB.') - 2*newpoint*x.'; %// Compute distance matrix
D = sqrt(D); %// Compute square root to complete calculation 

%// Sort the distances 
[d, ind] = sort(D, 2);

%// Get the indices of the closest distances
ind_closest = ind(:, 1:k);

endfunction

