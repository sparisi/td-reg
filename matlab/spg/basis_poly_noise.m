function Phi = basis_poly_noise(degree, dim, offset, state)
% Add noise to the observation of the state.
% The function is used both for state and state-action features. For this
% reason, the code is currently set for 2-dimensional states. Change line 
% 14 if you want to use larger states.

dimPhi = nmultichoosek(dim+1,degree);
if nargin == 3
    Phi = dimPhi;
    if ~offset
        Phi = Phi - 1;
    end
else
    state(1:2,:) = state(1:2,:) + 0.05*randn(2)*ones(size(state(1:2,:)))./min(max(abs(state(1:2,:)),0.1),200); % Noise
    
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
