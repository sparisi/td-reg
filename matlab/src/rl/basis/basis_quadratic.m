function Phi = basis_quadratic(dim, state)
% BASIS_QUADRATIC Computes full quadratic features: phi(s) = [1; s^1; s^2].
% It is analogous to BASIS_POLY, but faster.
%
%    INPUT
%     - dim   : dimension of the state
%     - state : (optional) [D x N] matrix of N states of size D to evaluate
%
%    OUTPUT
%     - Phi   : if a state is provided as input, the function 
%               returns the feature vectors representing it; 
%               otherwise it returns the number of features
%
% =========================================================================
% EXAMPLE
% basis_quadratic(3,[3,5,6]') = [1, 3, 5, 6, 9, 15, 18, 25, 30, 36]'

D = 2 * dim + dim * (dim - 1) / 2 + 1;

if nargin == 1
    Phi = D;
    return
end

[ds, N] = size(state);
assert(ds == dim, ...
    'State size is %d. Should be %d.', ds, dim)

combs = nchoose2(1:dim+1);
Phi = [ones(1,N)
    state
    state(combs(:,1),:) .* state(combs(:,2)-1,:)];
