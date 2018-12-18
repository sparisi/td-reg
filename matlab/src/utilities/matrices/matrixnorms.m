function norms = matrixnorms(M,idx)
% MATRIXNORMS Computes the norm of each row (IDX = 1) or column (IDX = 2)
% of a matrix M.

dims = [2 1];
norms = sqrt(sum(M.^2,dims(idx)));
