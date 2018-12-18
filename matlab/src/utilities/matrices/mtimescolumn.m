function C = mtimescolumn(A, B)
% MTIMESCOLUMN Multiplies each column of a matrix A by each row of a matrix
% B to obtain many 2d matrices. These matrices are then vectorized in a 2d
% matrix (one vectorization per column).
% It is equivalent to the following loop:
% >> for i = 1 : D
% >>     tmp = A(:,i) * B(:,i);
% >>     C(:,i) = tmp(:);
% >> end
%
%    INPUT
%     - A : [N x D] matrix
%     - B : [M x D] matrix
%
%    OUTPUT
%     - C : [N*M x D] matrix

N = size(A,1);
M = size(B,1);
C = reshape(bsxfun(@times,permute(A,[1 3 2]),permute(B,[3 1 2])),N*M,[]);
