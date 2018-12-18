function A = gae(data, V, gamma, lambda, prob_ratio)
% Computes generalized advantage estimates from potentially off-policy data.
% Data have to be ordered by episode: data.r must have first all samples 
% from the first episode, then all samples from the second, and so on.
% So you cannot use samples collected with COLLECT_SAMPLES2.
%
% Do not pass PROB_RATIO if data is on-policy.
% Truncate PROB_RATIO = min(1,PROB_RATIO) to use Retrace.
%
% =========================================================================
% REFERENCE
% J Schulman, P Moritz, S Levine, M Jordan, P Abbeel
% High-Dimensional Continuous Control Using Generalized Advantage Estimation
% ICLR (2017)
%
% R Munos, T Stepleton, Anna Harutyunyan, M G Bellemare
% Safe and efficient off-policy reinforcement learning
% NIPS (2016)

r = [data.r];
t = [data.t];
t(end+1) = 1;
A = zeros(size(V));

if nargin == 4 || isempty(prob_ratio), prob_ratio = ones(size(V)); end

for k = size(V,2) : -1 : 1
    if t(k+1) == 1 % Next state is a new episode init state
        A(k) = prob_ratio(k) * (r(k) - V(k));
    else
        A(k) = prob_ratio(k) * (r(k) + gamma*V(k+1) - V(k) + gamma*lambda*A(k+1));
    end
end
