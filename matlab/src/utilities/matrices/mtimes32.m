function C = mtimes32(A,B)
% MTIMES32 Multiplies a 3d matrix by a 2d matrix.
% It is equivalent to the following loop:
% >> C = zeros(size(A,1),size(A,3));
% >> for i = 1 : size(B,2)
% >>     C(:,i) = A(:,:,i) * B(:,i);
% >> end
%
%    INPUT
%     - A : [D1 x D2 x N] matrix
%     - B : [D2 x N] matrix
%
%    OUTPUT
%     - C : [D1 x N] matrix

C = bsxfun(@times,A,reshape(B,[1 size(B)]));
C = permute(sum(C,2),[1 3 2]);
