function Phi = basis_noisy(basis, sigma, varargin)
% Wrapper to add white noise to the observation of the state.

if nargin == 2
    Phi = basis();
else
    state = varargin{:};
    state = state + sigma*randn(size(state));
    Phi = basis(state);
end
