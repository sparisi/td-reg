function Phi = basis_fourier(n_feat, dim, B, offset, state)
% BASIS_FOURIER Random Fourier Radial Basis Functions. 
%
%    INPUT
%     - n_feat : number of total features
%     - dim    : size of the state
%     - B      : bandwidth (the authors suggest to use the average pairwise
%                distance between many samples (collected once before learning)
%     - offset : 1 to add an additional constant of value 1, 0 otherwise
%     - state  : (optional) [D x N] matrix of N states of size D to evaluate
%
%    OUTPUT
%     - Phi    : if a state is provided as input, the function 
%                returns the feature vectors representing it; 
%                otherwise it returns the number of features
% 
% =========================================================================
% REFERENCE
% https://arxiv.org/pdf/1703.02660.pdf

persistent P phase

if isempty(P)
    P = mymvnrnd(zeros(dim,n_feat),1);
    phase = unifrnd(-pi,pi,n_feat,1);
end

if nargin == 4
    Phi = n_feat + offset;
    return
end

Phi = sin(bsxfun(@plus, P'*(bsxfun(@times, state, 1./B)), phase));
if offset, Phi = [ones(1, size(state,2)); Phi]; end
