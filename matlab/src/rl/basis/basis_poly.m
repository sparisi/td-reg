function Phi = basis_poly(degree, dim, offset, state)
% BASIS_POLY Computes full polynomial features: phi(s) = s^0+s^1+s^2+...
% Since s is a vector, s^i denotes all the possible products of degree i
% between all elements of s, e.g., 
%
% s = (a, b, c)'
% s^3 = a^3 + b^3 + c^3 + a^2b + ab^2 + ac^2 + a^2c + b^2c + bc^2
%
%    INPUT
%     - degree : degree of the polynomial
%     - dim    : dimension of the state
%     - offset : 1 if you want to include the 0-degree component,
%                0 otherwise
%     - state  : (optional) [D x N] matrix of N states of size D to evaluate
%
%    OUTPUT
%     - Phi    : if a state is provided as input, the function 
%                returns the feature vectors representing it; 
%                otherwise it returns the number of features
%
% =========================================================================
% EXAMPLE
% basis_poly(2,3,1,[3,5,6]') = [1, 3, 5, 6, 9, 15, 18, 25, 30, 36]'

dimPhi = nmultichoosek(dim+1,degree);
if nargin == 3
    Phi = dimPhi;
    if ~offset
        Phi = Phi - 1;
    end
else
    assert(size(state,1) == dim, ...
        'State size is %d. Should be %d.', size(state,1),dim)
    nSamples = size(state,2);

    C = nmultichoosek([ones(1,nSamples); state], degree);
    Phi = permute(prod(C,2),[1,3,2]);

    if ~offset
        Phi(1,:) = [];
    end
end

return
