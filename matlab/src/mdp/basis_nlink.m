function Phi = basis_nlink(basis, varargin)
% Wrapper to transform the observation of the angle [theta] into [sin(theta); cos(theta)].

if nargin == 1
    Phi = basis();
else
    state = varargin{:};
    state = [cos(state(1:2:end,:)); sin(state(1:2:end,:)); state(2:2:end,:)];
    Phi = basis(state);
end