function d = kl_mvn2(p, q, phi)
% KL_MVN2 Computes the Kullback-Leibler divergence KL(P||Q) from distribution
% P to distribution Q, both Multivariate Normal with means linearly
% dependent on some features PHI.

s0 = p.Sigma;
s1 = q.Sigma;

dim = size(s0,1);

if ~p.no_bias
    m0 = p.A(:,1);
    K0 = p.A(:,2:end);
    m1 = q.A(:,1);
    K1 = q.A(:,2:end);
else
    m0 = zeros(dim,1);
    K0 = p.A;
    m1 = zeros(dim,1);
    K1 = q.A;
end    
    
const_diff = m0 - m1;
lin_diff = K0 - K1;
mu_states = mean(phi,2);
cov_states = cov(phi');

d = 0.5 * (trace(s1 \ s0) + const_diff' / s1 * const_diff - dim + logdet(s1,'chol') - logdet(s0,'chol'));

d = d + 0.5 * ( ...
    2 * mu_states' * lin_diff' / s1 * const_diff + ...
    mu_states' * lin_diff' / s1 * lin_diff * mu_states + ...
    trace(lin_diff' / s1 * lin_diff * cov_states) ...
    );
